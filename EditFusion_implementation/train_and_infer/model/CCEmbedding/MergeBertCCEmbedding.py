from train_and_infer.utils.tokenizer_util import (
    vocab_size,
    tokenizer,
    bert_path,
    edit_seq_tokens,
    bos_token_id,
    eos_token_id,
)
from torch.utils.data import Dataset
from transformers import AutoModel
from typing import Tuple
import pandas as pd
import torch.nn as nn
import torch
import os


script_path = os.path.dirname(os.path.abspath(__file__))

# 为 edit_seq 创建专门的词表映射，减小 embedding 层的体积
edit_token_values = list(edit_seq_tokens.values())
# <s>, </s>, <pad> 分别是句子开头,结尾和填充, edit_seq 也包含这些 token
edit_vocab_tokens = (
    [tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token] + edit_token_values
)
edit_token_ids = tokenizer.convert_tokens_to_ids(edit_vocab_tokens)

EDIT_ID_MAP = {original_id: i for i, original_id in enumerate(edit_token_ids)}
EDIT_VOCAB_SIZE = len(edit_vocab_tokens)
# 获取在新词表中的 padding_id
EDIT_PADDING_ID = EDIT_ID_MAP[tokenizer.pad_token_id]


class MergeBertCCEmbedding(nn.Module):
    """
    use pretrained model to embed the input explicitly aligned code change
    <s>origin_ids</s><pad><pad><pad><s>modified_ids</s><pad><pad><s>edit_seq</s><pad><pad><pad>
    """

    def __init__(self):
        super(MergeBertCCEmbedding, self).__init__()
        # 先拼接三个序列，然后使用预训练模型嵌入

        self.cc_embedding = AutoModel.from_pretrained(bert_path)

        # for param in self.cc_embedding.parameters():
        #     param.requires_grad = False   # 冻结预训练模型参数
        self.cc_embedding.resize_token_embeddings(
            vocab_size
        )  # 更新新增 token 后的词汇表大小

        self.edit_seq_embedding = nn.Embedding(
            EDIT_VOCAB_SIZE,
            self.cc_embedding.config.hidden_size,
            padding_idx=EDIT_PADDING_ID,
        )

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
        position_features, origin_ids, modified_ids, edit_seq_ids = feats
        # position_features 形状为 (batch_size, padded_seq_length（当前 batch 中最长序列的长度）, input_size)

        # 1. embed [origin, modified] with codebert
        code_ids = torch.cat([origin_ids, modified_ids], dim=2)  # N, L, 2*T

        flattened_code_ids = code_ids.contiguous().view(
            -1, code_ids.shape[-1]
        )  # N * L, 2*T

        bert_attention_mask = (flattened_code_ids != tokenizer.pad_token_id).type(
            torch.long
        )

        bert_embedding = self.cc_embedding(
            flattened_code_ids, attention_mask=bert_attention_mask
        ).last_hidden_state  # N * L, 2*T, embedding_dim

        code_cls_embedding = bert_embedding[:, 0, :]  # N * L, embedding_dim

        # 2. embed edit_seq with a separate embedding layer
        flattened_edit_seq_ids = edit_seq_ids.contiguous().view(
            -1, edit_seq_ids.shape[-1]
        )  # N * L, T

        edit_seq_embedding = self.edit_seq_embedding(
            flattened_edit_seq_ids
        )  # N * L, T, embedding_dim

        edit_seq_cls_embedding = edit_seq_embedding[:, 0, :]  # N * L, embedding_dim

        # 3. linear addition
        summed_embedding = code_cls_embedding + edit_seq_cls_embedding

        # 变成 batch_size * time_step * feature_size 的形式
        summed_embedding = summed_embedding.view(
            origin_ids.shape[0], origin_ids.shape[1], -1
        )

        # 和 position_features 拼接
        # todo 可以考虑加入特殊分割符
        catted_embedding = torch.cat((position_features, summed_embedding), dim=2)
        return catted_embedding

    @staticmethod
    def collate_fn(batch):
        batch.sort(
            key=lambda x: x[3], reverse=True
        )  # 根据序列长度降序排列，从长到短，为了后面 pack_padded_sequence
        position_features, triplets, labels, lengths, resolution_kinds = zip(
            *batch
        )  # 解构出 tuple，每个 tuple 的长度为 batch_size
        origin_ids, modified_ids, edit_seq_ids = zip(*triplets)

        # 下面 tuple 的长度为 batch_size，每个元素是一个 tensor，我们需要把这个 tensor 填充到同样的长度，得到一个 batch_size * max_length 的 tensor（因为batch_first=True）

        # pack_padded_sequence 会根据 lengths 来判断哪些是填充的，所以 padding_value 不需要指定
        position_features_padded = torch.nn.utils.rnn.pad_sequence(
            position_features, batch_first=True
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

        # ?不应该 padding，应该先 embedding 后再 pad 接着 pack 吗？这样送入 Embedding 时已经被 pad 了？岂不是有一个时间步是全 pad 的？
        # 下面这部分要经过预训练模型嵌入，预训练的 mask 的 pad_token_id 是 1，所以我们需要把 padding_value 设为 1
        origin_ids_padded = torch.nn.utils.rnn.pad_sequence(
            origin_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )  # padding_value 默认为 0
        modified_ids_padded = torch.nn.utils.rnn.pad_sequence(
            modified_ids, batch_first=True, padding_value=tokenizer.pad_token_id
        )
        edit_seq_ids_padded = torch.nn.utils.rnn.pad_sequence(
            edit_seq_ids, batch_first=True, padding_value=EDIT_PADDING_ID
        )

        return (
            (
                position_features_padded,
                origin_ids_padded,
                modified_ids_padded,
                edit_seq_ids_padded,
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
            position_features_df = target_edit_script[
                ["origin_start", "origin_end", "modified_start", "modified_end"]
            ]
            
            # 先转换为张量，避免在多进程加载时 pandas 和 numpy 的冲突
            position_features_tensor = torch.tensor(
                position_features_df.values, dtype=torch.float
            )

            # 使用 torch 计算长度特征
            origin_length = (position_features_tensor[:, 1] - position_features_tensor[:, 0]).unsqueeze(1)
            modified_length = (position_features_tensor[:, 3] - position_features_tensor[:, 2]).unsqueeze(1)
            length_difference = modified_length - origin_length

            # 拼接所有位置特征
            position_features = torch.cat(
                (position_features_tensor, origin_length, modified_length, length_difference), dim=1
            )
            
            # 抽取代码变更三个序列，注意写入时是以字符串形式写入的，所以读取时需要 eval
            _origin_processed_ids = list(
                map(eval, target_edit_script["origin_processed_ids"].values)
            )
            _modified_processed_ids = list(
                map(eval, target_edit_script["modified_processed_ids"].values)
            )
            _edit_seq_processed_ids = list(
                map(eval, target_edit_script["edit_seq_processed_ids"].values)
            )
            
            # 将 edit_seq_ids 映射到新的小词汇表
            _edit_seq_processed_ids_new = [
                [EDIT_ID_MAP[token_id] for token_id in seq]
                for seq in _edit_seq_processed_ids
            ]

            # 转换为张量
            labels_tensor = torch.tensor(labels, dtype=torch.float)
            origin_processed_ids = torch.tensor(_origin_processed_ids, dtype=torch.long)
            modified_processed_ids = torch.tensor(
                _modified_processed_ids, dtype=torch.long
            )
            edit_seq_processed_ids = torch.tensor(
                _edit_seq_processed_ids_new, dtype=torch.long
            )

            return (
                position_features,
                (origin_processed_ids, modified_processed_ids, edit_seq_processed_ids),
                labels_tensor,
                len(target_edit_script),
                resolution_kind,
            )


if __name__ == "__main__":
    print(MergeBertCCEmbedding().cc_embedding.get_input_embeddings())
    print(MergeBertCCEmbedding().cc_embedding)
