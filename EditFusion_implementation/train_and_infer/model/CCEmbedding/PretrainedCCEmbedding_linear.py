from train_and_infer.utils.tokenizer_util import vocab_size, pad_token_id
from torch.utils.data import Dataset
from transformers import AutoModel
from typing import Tuple
import pandas as pd
import torch.nn as nn
import torch
import os


script_path = os.path.dirname(os.path.abspath(__file__))
bert_path = os.path.join(script_path, "../../bert/CodeBERTa-small-v1")


class PretrainedCCEmbedding_linear(nn.Module):
    """
    use pretrained model to embed the input explicitly aligned code change
    """

    def __init__(self):
        super(PretrainedCCEmbedding_linear, self).__init__()
        # 先拼接三个序列，然后使用预训练模型嵌入
        self.cc_embedding = AutoModel.from_pretrained(
            bert_path
        )  # 使用 Roberta model with no head
        # for param in self.cc_embedding.parameters():
        #     param.requires_grad = False   # 冻结预训练模型参数
        self.cc_embedding.resize_token_embeddings(
            vocab_size
        )  # 更新新增 token 后的词汇表大小
        self.embed_dim = self.cc_embedding.config.hidden_size
        self.linear_fusion = nn.Linear(
            self.embed_dim * 3, self.embed_dim
        )  # 线性变换，将三个序列的嵌入拼接后降维到 embed_dim

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

    def forward(self, feats: tuple):
        position_features, origin_ids, modified_ids, edit_seq_ids = feats

        # 分别嵌入三个序列
        origin_ids = origin_ids.view(
            -1, origin_ids.shape[-1]
        )  # N * time_steps, token_length
        modified_ids = modified_ids.view(
            -1, modified_ids.shape[-1]
        )  # N * time_steps, token_length
        edit_seq_ids = edit_seq_ids.view(
            -1, edit_seq_ids.shape[-1]
        )  # N * time_steps, token_length

        # attention_mask
        attention_mask = (origin_ids != pad_token_id).type(
            torch.long
        )  # 3 个序列的 mask 是一样的，所以只取一个就行

        origin_embeds = self.cc_embedding(
            origin_ids, attention_mask=attention_mask
        ).last_hidden_state[
            :, 0, :
        ]  # N * time_steps, embed_dim
        modified_embeds = self.cc_embedding(
            modified_ids, attention_mask=attention_mask
        ).last_hidden_state[
            :, 0, :
        ]  # N * time_steps, embed_dim
        edit_seq_embeds = self.cc_embedding(
            edit_seq_ids, attention_mask=attention_mask
        ).last_hidden_state[
            :, 0, :
        ]  # N * time_steps, embed_dim

        # 拼接嵌入
        concatenated_embeds = torch.cat(
            [origin_embeds, modified_embeds, edit_seq_embeds], dim=-1
        )  # N * time_steps, 3 * embed_dim

        # 简单线性变换
        fused_embedding = self.linear_fusion(
            concatenated_embeds
        )  # N * time_steps, output_dim

        # 调整形状以匹配时间步长 (time_steps) 和 batch 结构
        fused_embedding = fused_embedding.view(
            position_features.shape[0], position_features.shape[1], -1
        )  # batch_size, time_steps, output_dim

        # 和 position_features 拼接
        catted_embedding = torch.cat(
            (position_features, fused_embedding), dim=2
        )  # batch_size, time_steps, feature_dim + output_dim
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
            origin_ids, batch_first=True, padding_value=1
        )  # padding_value 默认为 0
        modified_ids_padded = torch.nn.utils.rnn.pad_sequence(
            modified_ids, batch_first=True, padding_value=1
        )
        edit_seq_ids_padded = torch.nn.utils.rnn.pad_sequence(
            edit_seq_ids, batch_first=True, padding_value=1
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
            _origin_processed_ids = list(
                map(eval, target_edit_script["origin_processed_ids"].values)
            )
            _modified_processed_ids = list(
                map(eval, target_edit_script["modified_processed_ids"].values)
            )
            _edit_seq_processed_ids = list(
                map(eval, target_edit_script["edit_seq_processed_ids"].values)
            )

            # 转换为张量
            position_features = torch.tensor(
                position_features.values, dtype=torch.float
            )
            labels_tensor = torch.tensor(labels, dtype=torch.float)
            origin_processed_ids = torch.tensor(_origin_processed_ids, dtype=torch.long)
            modified_processed_ids = torch.tensor(
                _modified_processed_ids, dtype=torch.long
            )
            edit_seq_processed_ids = torch.tensor(
                _edit_seq_processed_ids, dtype=torch.long
            )

            return (
                position_features,
                (origin_processed_ids, modified_processed_ids, edit_seq_processed_ids),
                labels_tensor,
                len(target_edit_script),
                resolution_kind,
            )


if __name__ == "__main__":
    print(PretrainedCCEmbedding().cc_embedding.get_input_embeddings())
    print(PretrainedCCEmbedding().cc_embedding)
