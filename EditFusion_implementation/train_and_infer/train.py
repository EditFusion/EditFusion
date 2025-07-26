from collections import defaultdict
from typing import List
import pandas as pd
from pathlib import Path
import torch
import torch.optim as optim
from tqdm import tqdm
import os
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset

from .model.LSTM_model import LSTMClassifier, model_params
from .utils.tokenizer_util import pad_token_id
from .utils.util import ObjectDict

training_param = ObjectDict({
    'TRAIN_SPLIT': 0.8,
    'learning_rate': 5e-6,
    'epochs': 5,
    'batch_size': 2,
})
# Build Dataset
class EditScriptDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.block_ids = dataframe['block_id'].unique()

    def __len__(self):
        return len(self.block_ids)

    def __getitem__(self, idx):
        block_id = self.block_ids[idx]
        target_edit_script = self.dataframe[self.dataframe['block_id'] == block_id] # Get all edit scripts for current block_id
        resolution_kind = target_edit_script['resolution_kind'].iloc[0]

        # Separate labels
        labels = target_edit_script['accept'].values

        # Extract position and length features
        position_features = target_edit_script[['origin_start', 'origin_end', 'modified_start', 'modified_end']].copy()
        position_features['origin_length'] = position_features['origin_end'] - position_features['origin_start']
        position_features['modified_length'] = position_features['modified_end'] - position_features['modified_start']
        position_features['length_difference'] = position_features['modified_length'] - position_features['origin_length']
        # Extract three code change sequences; note: written as string, so need eval when reading
        _origin_processed_ids = list(map(eval, target_edit_script['origin_processed_ids'].values))
        _modified_processed_ids = list(map(eval, target_edit_script['modified_processed_ids'].values))
        _edit_seq_processed_ids = list(map(eval, target_edit_script['edit_seq_processed_ids'].values))

        # Convert to tensor
        position_features = torch.tensor(position_features.values, dtype=torch.float)
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        origin_processed_ids = torch.tensor(_origin_processed_ids, dtype=torch.long)
        modified_processed_ids = torch.tensor(_modified_processed_ids, dtype=torch.long)
        edit_seq_processed_ids = torch.tensor(_edit_seq_processed_ids, dtype=torch.long)

        return position_features, (origin_processed_ids, modified_processed_ids, edit_seq_processed_ids), labels_tensor, len(target_edit_script), resolution_kind

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)  # Sort by sequence length (descending) for pack_padded_sequence
    position_features, triplets, labels, lengths, resolution_kinds = zip(*batch)
    origin_ids, modified_ids, edit_seq_ids = zip(*triplets)

    # Each tuple element is a tensor; pad to same length for batch (batch_first=True)
    position_features_padded = torch.nn.utils.rnn.pad_sequence(position_features, batch_first=True)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    origin_ids_padded = torch.nn.utils.rnn.pad_sequence(origin_ids, batch_first=True)
    modified_ids_padded = torch.nn.utils.rnn.pad_sequence(modified_ids, batch_first=True)
    edit_seq_ids_padded = torch.nn.utils.rnn.pad_sequence(edit_seq_ids, batch_first=True)

    return position_features_padded, (origin_ids_padded, modified_ids_padded, edit_seq_ids_padded), labels_padded, torch.tensor(lengths), resolution_kinds


if __name__ == '__main__':
    # Set random seed
    torch.manual_seed(42)

    # Load data
    script_path = Path(os.path.dirname(os.path.abspath(__file__)))
    # NOTE: The following file name is in Chinese. Consider renaming to English for public release.
    dataset = pd.read_csv(script_path / 'data' / 'CC嵌入数据集_tokenlen32.csv')

    # Create DataSet
    LSTM_dataset = EditScriptDataset(dataset)
    # LSTM_dataset = Subset(LSTM_dataset, range(5000))

    # Calculate train/test split sizes
    total_size = len(LSTM_dataset)
    train_size = int(total_size * training_param.TRAIN_SPLIT)
    test_size = total_size - train_size

    # Randomly split dataset
    train_dataset, test_dataset = random_split(LSTM_dataset, [train_size, test_size])

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=training_param.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=training_param.batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize model
    model = LSTMClassifier(**model_params)

    # Move model to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=training_param.learning_rate)

    # Training loop
    for epoch in range(training_param.epochs):
        torch.cuda.empty_cache()
        model.train()
        total_loss = 0
        for position_features, (origin_ids, modified_ids, edit_seq_ids), labels, lengths, _ in tqdm(train_loader, dynamic_ncols=True, desc=f'Epoch {epoch + 1}/{training_param.epochs}'):
            # position_features shape: (batch_size, padded_seq_length, input_size)

            # Move data to device
            position_features, origin_ids, modified_ids, edit_seq_ids, labels = position_features.to(device), origin_ids.to(device), modified_ids.to(device), edit_seq_ids.to(device), labels.to(device)

            # Forward pass
            outputs = model(position_features, origin_ids, modified_ids, edit_seq_ids, lengths)
            outputs = outputs.squeeze(2)

            # Use mask to ignore padded outputs
            mask = torch.arange(outputs.size(1)).expand(len(lengths), outputs.size(1)) < lengths.unsqueeze(1)
            mask = mask.to(device)
            outputs_selected = outputs.masked_select(mask)
            labels_selected = labels.masked_select(mask)
            loss = criterion(outputs_selected, labels_selected)

            # Zero gradients
            optimizer.zero_grad()
            # Backward and optimize
            loss.backward()
            total_loss += loss
            optimizer.step()
            torch.cuda.empty_cache()

        train_loss = total_loss / len(train_loader)
        total_loss = 0  # Reset for validation loop
        print(f'Epoch {epoch + 1}/{training_param.epochs} | Train Loss: {train_loss}')

        # Validation loop
        with torch.inference_mode():
            correct_num = 0
            total_num = 0
            kind_counter = defaultdict(int)
            kind_correct_counter = defaultdict(int)
            model.eval()
            for position_features, (origin_ids, modified_ids, edit_seq_ids), labels, lengths, resolution_kinds in tqdm(test_loader, dynamic_ncols=True, desc=f'test samples in {epoch + 1}/{training_param.epochs}'):
                curr_batch_size = len(position_features)
                # Move data to device
                position_features, origin_ids, modified_ids, edit_seq_ids, labels = position_features.to(device), origin_ids.to(device), modified_ids.to(device), edit_seq_ids.to(device), labels.to(device)

                # Forward pass
                outputs = model(position_features, origin_ids, modified_ids, edit_seq_ids, lengths)
                outputs = outputs.squeeze(2)

                # Only compare non-padded parts
                mask = torch.arange(outputs.size(1)).expand(len(lengths), outputs.size(1)) < lengths.unsqueeze(1)
                mask = mask.to(device)
                outputs_selected = outputs.masked_select(mask)
                labels_selected = labels.masked_select(mask)
                loss = criterion(outputs_selected, labels_selected)
                total_loss += loss

                # Calculate accuracy
                outputs = outputs * mask
                labels = labels * mask
                outputs = outputs.round().int()
                labels = labels.int()

                for i in range(curr_batch_size):
                    kind_counter[resolution_kinds[i]] += 1
                    if torch.equal(outputs[i], labels[i]):
                        correct_num += 1
                        kind_correct_counter[resolution_kinds[i]] += 1
                total_num += curr_batch_size

            test_loss = total_loss / len(test_loader)
            print(f'Epoch {epoch + 1}/{training_param.epochs} | Test Loss: {test_loss}')
            print(f'Accuracy: {round(correct_num / total_num * 100, 2)}%')
            for kind in kind_counter.keys():
                print(f'Accuracy on {kind}: {round(kind_correct_counter[kind] / kind_counter[kind] * 100, 2)}%, {kind_correct_counter[kind]}/{kind_counter[kind]}')
            print('\n' * 2)

    # Print training parameters
    print('Training parameters:')
    print(training_param)

    # Save model
    torch.save(model.state_dict(), script_path / 'data' / f'Bert_embedding_bs{training_param.batch_size}_lr{training_param.learning_rate}.pth')