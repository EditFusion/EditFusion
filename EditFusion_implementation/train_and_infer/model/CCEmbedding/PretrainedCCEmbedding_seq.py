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
        self.cc_embedding = AutoModel.from_pretrained(
            bert_path
        )
        self.cc_embedding.resize_token_embeddings(
            vocab_size
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

    def forward(self, feats: Tuple):
        position_features, cc_ids = feats

        flattened_ids = cc_ids.contiguous().view(
            -1, cc_ids.shape[-1]
        )
        bert_attention_mask = (flattened_ids != tokenizer.pad_token_id).type(torch.long)
        bert_embedding = self.cc_embedding(
            flattened_ids, attention_mask=bert_attention_mask
        ).last_hidden_state
        cls_embedding = bert_embedding[:, 0, :]

        cls_embedding = cls_embedding.view(cc_ids.shape[0], cc_ids.shape[1], -1)

        catted_embedding = torch.cat((position_features, cls_embedding), dim=2)
        return catted_embedding

    @staticmethod
    def collate_fn(batch):
        batch.sort(
            key=lambda x: x[3], reverse=True
        )
        position_features, cc_ids, labels, lengths, resolution_kinds = zip(
            *batch
        )

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
                map(eval, target_edit_script["origin_ids_truncated"].values)
            )
            _modified_processed_ids = list(
                map(eval, target_edit_script["modified_ids_truncated"].values)
            )
            _edit_seq_processed_ids = list(
                map(eval, target_edit_script["edit_seq_ids_truncated"].values)
            )

            def merge_seq_ids(
                ids: Tuple[Tuple[int]],
                bos_token_id: int,
                sep_token_id: int,
                pad_token_id: int,
                max_length: int,
            ) -> torch.Tensor:
                """
                1. Concatenate all tokens into a single list at the Python level.
                2. Convert to a PyTorch Tensor at once to avoid frequently creating and concatenating small tensors in a loop.
                Args:
                    ids: A nested tuple with a shape like (3, seq_len, some_token_length).
                    bos_token_id, sep_token_id, pad_token_id: Corresponding to the tokenizer's BOS, SEP, PAD.
                    max_length: The maximum sequence length to truncate or pad to.

                Returns:
                    A LongTensor with a shape of (seq_len, max_length).
                """
                seq_len = len(ids[0])

                output = []

                for i in range(seq_len):
                    row_tokens = [bos_token_id]
                    for single_ids in ids:
                        row_tokens.extend(single_ids[i])
                        row_tokens.append(sep_token_id)

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
