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
        self.cc_embedding = AutoModel.from_pretrained(
            bert_path
        )
        self.cc_embedding.resize_token_embeddings(
            vocab_size
        )
        self.embed_dim = self.cc_embedding.config.hidden_size
        self.linear_fusion = nn.Linear(
            self.embed_dim * 3, self.embed_dim
        )

    def get_dataset(self, dataset_path):
        """
        Load multiple CSV files in chunks and merge them into a single Dataset.
        """
        if os.path.isdir(dataset_path):
            all_files = [
                os.path.join(dataset_path, file)
                for file in os.listdir(dataset_path)
                if file.endswith(".csv")
            ]

            chunk_size = 10000
            chunks = []
            for file in all_files:
                for chunk in pd.read_csv(file, chunksize=chunk_size):
                    chunks.append(chunk)
            dataset_df = pd.concat(chunks, ignore_index=True)
        else:
            dataset_df = pd.read_csv(dataset_path)

        return self.LTREPretrainedDataset(dataset_df)

    def forward(self, feats: tuple):
        position_features, origin_ids, modified_ids, edit_seq_ids = feats

        origin_ids = origin_ids.view(
            -1, origin_ids.shape[-1]
        )
        modified_ids = modified_ids.view(
            -1, modified_ids.shape[-1]
        )
        edit_seq_ids = edit_seq_ids.view(
            -1, edit_seq_ids.shape[-1]
        )

        attention_mask = (origin_ids != pad_token_id).type(
            torch.long
        )

        origin_embeds = self.cc_embedding(
            origin_ids, attention_mask=attention_mask
        ).last_hidden_state[
            :, 0, :
        ]
        modified_embeds = self.cc_embedding(
            modified_ids, attention_mask=attention_mask
        ).last_hidden_state[
            :, 0, :
        ]
        edit_seq_embeds = self.cc_embedding(
            edit_seq_ids, attention_mask=attention_mask
        ).last_hidden_state[
            :, 0, :
        ]

        concatenated_embeds = torch.cat(
            [origin_embeds, modified_embeds, edit_seq_embeds], dim=-1
        )

        fused_embedding = self.linear_fusion(
            concatenated_embeds
        )

        fused_embedding = fused_embedding.view(
            position_features.shape[0], position_features.shape[1], -1
        )

        catted_embedding = torch.cat(
            (position_features, fused_embedding), dim=2
        )
        return catted_embedding

    @staticmethod
    def collate_fn(batch):
        batch.sort(
            key=lambda x: x[3], reverse=True
        )
        position_features, triplets, labels, lengths, resolution_kinds = zip(
            *batch
        )
        origin_ids, modified_ids, edit_seq_ids = zip(*triplets)

        position_features_padded = torch.nn.utils.rnn.pad_sequence(
            position_features, batch_first=True
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

        origin_ids_padded = torch.nn.utils.rnn.pad_sequence(
            origin_ids, batch_first=True, padding_value=1
        )
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

            labels = target_edit_script["accept"].values

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
            _origin_processed_ids = list(
                map(eval, target_edit_script["origin_processed_ids"].values)
            )
            _modified_processed_ids = list(
                map(eval, target_edit_script["modified_processed_ids"].values)
            )
            _edit_seq_processed_ids = list(
                map(eval, target_edit_script["edit_seq_processed_ids"].values)
            )

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
