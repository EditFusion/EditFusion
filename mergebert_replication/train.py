# This script will be used to fine-tune the MergeBERT model.

import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn as nn
from tqdm import tqdm

from model import MergeBERTModel

# Constants
DATA_DIR = "/home/foril/projects/EditFusion/mergebert_replication/data/"
MODEL_PATH = "/home/foril/projects/EditFusion/mergebert_replication/CodeBERTa-small-v1/"
SAVED_MODEL_DIR = "/home/foril/projects/EditFusion/mergebert_replication/saved_models/"
NUM_LABELS = 5 # A, B, O, AB, BA
EPOCHS = 3
BATCH_SIZE = 16 # Using a smaller batch size due to potential memory constraints
LEARNING_RATE = 5e-5

class MergeDataset(Dataset):
    """Custom PyTorch Dataset for the merge conflict data."""
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "a_o_ids": torch.tensor(sample['a_o_aligned'], dtype=torch.long),
            "o_a_ids": torch.tensor(sample['o_a_aligned'], dtype=torch.long),
            "b_o_ids": torch.tensor(sample['b_o_aligned'], dtype=torch.long),
            "o_b_ids": torch.tensor(sample['o_b_aligned'], dtype=torch.long),
            "labels": torch.tensor(sample['label'], dtype=torch.long)
        }

def compute_metrics(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(data_loader, desc="Training"):
        optimizer.zero_grad()
        
        a_o_ids = batch['a_o_ids'].to(device)
        o_a_ids = batch['o_a_ids'].to(device)
        b_o_ids = batch['b_o_ids'].to(device)
        o_b_ids = batch['o_b_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(a_o_ids, o_a_ids, b_o_ids, o_b_ids, labels=labels)
        loss = outputs['loss']
        if device.type == 'cuda' and torch.cuda.device_count() > 1:
            loss = loss.mean()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(data_loader)

def eval_epoch(model, data_loader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            a_o_ids = batch['a_o_ids'].to(device)
            o_a_ids = batch['o_a_ids'].to(device)
            b_o_ids = batch['b_o_ids'].to(device)
            o_b_ids = batch['o_b_ids'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(a_o_ids, o_a_ids, b_o_ids, o_b_ids, labels=labels)
            loss = outputs['loss']
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                loss = loss.mean()
            logits = outputs['logits']

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    metrics = compute_metrics(all_preds, all_labels)
    return avg_loss, metrics

def main():
    """Main function to run the training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = MergeDataset(os.path.join(DATA_DIR, "train.json"))
    val_dataset = MergeDataset(os.path.join(DATA_DIR, "validation.json"))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = MergeBERTModel(encoder_path=MODEL_PATH, num_labels=NUM_LABELS)
    model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_val_accuracy = 0

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"Train loss: {train_loss:.4f}")

        val_loss, metrics = eval_epoch(model, val_loader, device)
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation metrics: {metrics}")

        if metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = metrics['accuracy']
            # If using DataParallel, the model is wrapped in a module
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), os.path.join(SAVED_MODEL_DIR, 'mergebert.pt'))
            print(f"Best model saved to {SAVED_MODEL_DIR}")

if __name__ == "__main__":
    main()
