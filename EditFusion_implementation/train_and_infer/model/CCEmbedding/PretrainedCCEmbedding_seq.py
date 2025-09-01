from train_and_infer.utils.tokenizer_util import vocab_size, tokenizer, bert_path
from torch.utils.data import Dataset
from transformers import AutoModel
from typing import Tuple
import pandas as pd
import torch.nn as nn
import torch
import os


script_path = os.path.dirname(os.path.abspath(__file__))
MAX_TOKEN_LEN = 200


class PretrainedCCEmbeddingSeq(nn.Module):
    """
    use pretrained model to embed the input explicitly aligned code change
    """

    def __init__(self):
        super(PretrainedCCEmbeddingSeq, self).__init__()
        # 先拼接三个序列，然后使用预训练模型嵌入
        self.cc_embedding = AutoModel.from_pretrained(
            bert_path
        )  # 使用 Roberta model with no head
        # for param in self.cc_embedding.parameters():
        #     param.requires_grad = False   # 冻结预训练模型参数
        self.cc_embedding.resize_token_embeddings(
            vocab_size
        )  # 更新新增 token 后的词汇表大小

    def get_dataset(self, dataset_path):
        """
        按块加载多个 CSV 文件，并合并为一个 Dataset
        """
        if os.path.isdir(dataset_path):
            all_files = [
                os.path.join(dataset_path, file)
                for file in os.listdir(dataset_path)
                if file.endswith(".csv")
            ]

            # 按块加载并处理
            chunk_size = 10000  # 每次加载 10000 行
            chunks = []
            for file in all_files:
                for chunk in pd.read_csv(file, chunksize=chunk_size):
                    # 在这里可以对每块数据进行预处理
                    chunks.append(chunk)
            dataset_df = pd.concat(chunks, ignore_index=True)
        else:
            dataset_df = pd.read_csv(dataset_path)

        return self.LTREPretrainedDataset(dataset_df)

    def forward(self, feats: Tuple):
        position_features, cc_ids = feats
        # position_features 形状为 (batch_size, padded_seq_length（当前 batch 中最长序列的长度）, input_size)
        # cc_ids 为 N, L, 200，然后使用预训练模型嵌入为 N，L，200，embedding_dim 的张量

        # 展平
        flattened_ids = cc_ids.contiguous().view(
            -1, cc_ids.shape[-1]
        )  # N * L, 200
        bert_attention_mask = (flattened_ids != tokenizer.pad_token_id).type(torch.long)
        bert_embedding = self.cc_embedding(
            flattened_ids, attention_mask=bert_attention_mask
        ).last_hidden_state  # N * L, 200, embedding_dim
        # 取出 第一个 <s> token 的 embedding
        cls_embedding = bert_embedding[:, 0, :]  # N * L, embedding_dim

        # 变成 batch_size * time_step * feature_size 的形式
        cls_embedding = cls_embedding.view(cc_ids.shape[0], cc_ids.shape[1], -1)

        # 和 position_features 拼接
        catted_embedding = torch.cat((position_features, cls_embedding), dim=2)
        return catted_embedding

    @staticmethod
    def collate_fn(batch):
        batch.sort(
            key=lambda x: x[3], reverse=True
        )  # 根据序列长度降序排列，从长到短，为了后面 pack_padded_sequence
        position_features, cc_ids, labels, lengths, resolution_kinds = zip(
            *batch
        )  # 解构出 tuple，每个 tuple 的长度为 batch_size

        # 下面 tuple 的长度为 batch_size，每个元素是一个 tensor，我们需要把这个 tensor 填充到同样的长度，得到一个 batch_size * max_length 的 tensor（因为batch_first=True）

        # pack_padded_sequence 会根据 lengths 来判断哪些是填充的，所以 padding_value 不需要指定
        position_features_padded = torch.nn.utils.rnn.pad_sequence(
            position_features, batch_first=True
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

        cc_ids_padded = torch.nn.utils.rnn.pad_sequence(
            cc_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )

        return (
            (
                position_features_padded,
                cc_ids_padded,
            ),
            labels_padded,
            torch.tensor(lengths),
            resolution_kinds,
        )

    class LTREPretrainedDataset(Dataset):
        def __init__(self, dataframe):
            self.grouped_data = dataframe.groupby("block_id")
            self.block_ids = list(self.grouped_data.groups.keys())

        def __len__(self):
            return len(self.block_ids)

        def __getitem__(self, idx):
            block_id = self.block_ids[idx]
            target_edit_script = self.grouped_data.get_group(block_id)
            resolution_kind = target_edit_script["resolution_kind"].iloc[0]

            # 分离标签
            labels = target_edit_script["accept"].values

            # 抽取位置长度特征
            position_features = target_edit_script[
                ["origin_start", "origin_end", "modified_start", "modified_end"]
            ].copy()
            position_features["origin_length"] = (
                position_features["origin_end"] - position_features["origin_start"]
            )
            position_features["modified_length"] = (
                position_features["modified_end"] - position_features["modified_start"]
            )
            position_features["length_difference"] = (
                position_features["modified_length"]
                - position_features["origin_length"]
            )
            # 抽取代码变更三个序列，注意写入时是以字符串形式写入的，所以读取时需要 eval
            # 长度不一定相等
            _origin_processed_ids = list(
                map(eval, target_edit_script["origin_ids_truncated"].values)
            )
            _modified_processed_ids = list(
                map(eval, target_edit_script["modified_ids_truncated"].values)
            )
            _edit_seq_processed_ids = list(
                map(eval, target_edit_script["edit_seq_ids_truncated"].values)
            )

            # 以下是对序列进行处理，将其拼接为一个序列
            def merge_seq_ids(
                ids: Tuple[Tuple[int]],
                bos_token_id: int,
                sep_token_id: int,
                pad_token_id: int,
                max_length: int,
            ) -> torch.Tensor:
                """
                1. 在 Python 层面先拼接所有 token 到一个 list。
                2. 再一次性转换为 PyTorch Tensor，避免在循环中频繁地创建小张量并 cat。
                参数：
                ids: 形状类似 (3, seq_len, some_token_length) 的嵌套元组
                bos_token_id, sep_token_id, pad_token_id: 分别对应 tokenizer 的 BOS、SEP、PAD
                max_length: 截断或补齐到的最大序列长度

                返回：
                形状为 (seq_len, max_length) 的 LongTensor
                """
                seq_len = len(ids[0])

                # 存放每个时间步的 token 序列
                output = []

                # 遍历每个时间步
                for i in range(seq_len):
                    # 构建该时间步的 token list
                    row_tokens = [bos_token_id]
                    for single_ids in ids:
                        row_tokens.extend(single_ids[i])  # 拼接原始 token
                        row_tokens.append(sep_token_id)  # 拼接 SEP

                    # 截断或补齐到 max_length
                    if len(row_tokens) > max_length:
                        row_tokens = row_tokens[:max_length]
                    else:
                        row_tokens += [pad_token_id] * (max_length - len(row_tokens))

                    output.append(row_tokens)

                return torch.tensor(output, dtype=torch.long)

            cc_ids = merge_seq_ids(
                (
                    _origin_processed_ids,
                    _modified_processed_ids,
                    _edit_seq_processed_ids,
                ),
                bos_token_id=tokenizer.bos_token_id,
                sep_token_id=tokenizer.sep_token_id,
                pad_token_id=tokenizer.pad_token_id,
                max_length=MAX_TOKEN_LEN
            )

            # 转换为张量
            position_features = torch.tensor(
                position_features.values, dtype=torch.float
            )
            labels_tensor = torch.tensor(labels, dtype=torch.float)
            return (
                position_features,
                cc_ids,
                labels_tensor,
                len(target_edit_script),
                resolution_kind,
            )


if __name__ == "__main__":
    print(PretrainedCCEmbedding().cc_embedding.get_input_embeddings())
    print(PretrainedCCEmbedding().cc_embedding)
