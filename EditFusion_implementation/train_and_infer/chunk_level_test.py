import pandas as pd
from .model.LSTM_model import LSTMClassifier
from .params import model_params
import torch
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, Subset
from .utils.model_util import load_model_param

model = LSTMClassifier(**model_params)
script_path = Path(os.path.dirname(os.path.abspath(__file__)))
model_path = script_path / "data" / "tmp.pth"

model, device = load_model_param(model, model_path)
model.eval()

dataset_path = script_path / "data" / "CC嵌入数据集_tokenlen32.csv"
dataset = model.CCEmbedding_class.get_dataset(dataset_path)



data_loader = DataLoader(
    dataset, batch_size=4, shuffle=False, collate_fn=model.CCEmbedding_class.collate_fn
)


def mask_output(outputs, labels, lengths):
    """Masks the output and labels based on the actual sequence lengths."""
    # todo This function is duplicated from LSTMTrainer, refactor to a single location.
    outputs = outputs[
        :, : max(lengths), :
    ]

    outputs = outputs.squeeze(2)

    mask = torch.arange(outputs.size(1)).expand(
        len(lengths), outputs.size(1)
    ) < lengths.unsqueeze(1)
    mask = mask.to(device)
    outputs_selected = outputs.masked_select(mask)
    labels_selected = labels.masked_select(mask)
    return outputs_selected, labels_selected, mask


def accuracy_on_dataset(data_loader):
    """Calculate and print the accuracy of the model on the given dataset."""
    # inference
    with torch.inference_mode():
        correct_num = 0
        total_num = 0
        kind_counter = defaultdict(int)
        kind_correct_counter = defaultdict(int)
        model.eval()
        for loaded_feats, labels, lengths, resolution_kinds in tqdm(
            data_loader, dynamic_ncols=True, desc=f"counting chunk accuracy"
        ):
            labels = labels.to(device)
            if isinstance(loaded_feats, tuple):
                loaded_feats = tuple(f.to(device) for f in loaded_feats)

            # todo Verify the correctness of the tensor shapes here.
            outputs = model(
                loaded_feats, lengths
            )

            outputs_selected, labels_selected, _ = mask_output(outputs, labels, lengths)

            curr_batch_size = labels.shape[0]
            for i in range(curr_batch_size):
                kind_counter[resolution_kinds[i]] += 1
                if torch.equal(
                    outputs[i], labels[i]
                ):
                    correct_num += 1
                    kind_correct_counter[resolution_kinds[i]] += 1
            total_num += curr_batch_size
        print(f"Accuracy: {round(correct_num / total_num * 100, 2)}%")

        for kind in kind_counter.keys():
            print(
                f"Accuracy on {kind}: {round(kind_correct_counter[kind] / kind_counter[kind] * 100, 2)}%, {kind_correct_counter[kind]}/{kind_counter[kind]}"
            )


accuracy_on_dataset(data_loader)
