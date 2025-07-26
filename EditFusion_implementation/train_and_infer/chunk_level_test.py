import pandas as pd
from .model.LSTM_model import LSTMClassifier, model_params
import torch
import os
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader
from .train import collate_fn, EditScriptDataset

 # Load state dict
 # First, recreate a model instance with the same architecture as the original
model = LSTMClassifier(**model_params)
script_path = Path(os.path.dirname(os.path.abspath(__file__)))
model.load_state_dict(torch.load(script_path / 'data' / 'Bert_embedding_bs2_lr5e-06.pth'))
model.eval()  # Switch to evaluation mode

# Move model to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
print(f'Using {device} device')

# Load dataset
dataset = pd.read_csv(script_path / 'data' / 'CC_embedding_dataset_tokenlen32.csv')
data_loader = DataLoader(EditScriptDataset(dataset), batch_size=4, shuffle=False, collate_fn=collate_fn)

def accuracy_on_dataset(data_loader):
    # Inference
    with torch.inference_mode():
        correct_num = 0
        total_num = 0
        kind_counter = defaultdict(int)
        kind_correct_counter = defaultdict(int)
        model.eval()
        for position_features, (origin_ids, modified_ids, edit_seq_ids), labels, lengths, resolution_kinds  in tqdm(data_loader, dynamic_ncols=True, desc=f'counting chunk accuracy'):
            curr_batch_size = position_features.shape[0]
            # position_features shape: (batch_size, padded_seq_length (max length in current batch), input_size)

            # Move data to GPU
            position_features, origin_ids, modified_ids, edit_seq_ids, labels = position_features.to(device), origin_ids.to(device), modified_ids.to(device), edit_seq_ids.to(device), labels.to(device)

            # Forward pass
            outputs = model(position_features, origin_ids, modified_ids, edit_seq_ids, lengths)    # pack_padded_sequence then pad_packed_sequence
            # Calculate loss
            # Output and label shapes must match, so expand labels as needed
            outputs = outputs.squeeze(2)
            # Convert to 0/1
            outputs = outputs.round().int()
            labels = labels.int()

            # Only compare non-padding part of each sample
            mask = torch.arange(outputs.size(1)).expand(len(lengths), outputs.size(1)) < lengths.unsqueeze(1)
            mask = mask.to(device)
            # Calculate accuracy
            outputs = outputs * mask
            labels = labels * mask 

            # If outputs and labels are equal for each sample, prediction is correct
            for i in range(curr_batch_size):
                kind_counter[resolution_kinds[i]] += 1
                if torch.equal(outputs[i], labels[i]):      # Only compare labels; empty lines filtered during dataset creation
                    correct_num += 1
                    kind_correct_counter[resolution_kinds[i]] += 1
            total_num += curr_batch_size
        print(f'Accuracy: {round(correct_num / total_num * 100, 2)}%')

        for kind in kind_counter.keys():
            print(f'Accuracy on {kind}: {round(kind_correct_counter[kind] / kind_counter[kind] * 100, 2)}%, {kind_correct_counter[kind]}/{kind_counter[kind]}')

accuracy_on_dataset(data_loader)