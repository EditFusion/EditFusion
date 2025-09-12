import datetime
import os
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from model import MergeBERTModel

import logging

# Constants
DATA_DIR = "/home/foril/projects/EditFusion/mergebert_replication/data/"
MODEL_PATH = "/home/foril/projects/EditFusion/mergebert_replication/CodeBERTa-small-v1/"
SAVED_MODEL_DIR = "/home/foril/projects/EditFusion/mergebert_replication/saved_models/"
NUM_LABELS = 9
EPOCHS = 3 # Run for 3 epochs as requested
BATCH_SIZE = 16
LEARNING_RATE = 5e-5

LABEL_MAP = {
    "select_a": 0,
    "select_b": 1,
    "select_o": 2,
    "concat_ab": 3,
    "concat_ba": 4,
    "select_a_del_o": 5,
    "select_b_del_o": 6,
    "concat_ab_del_o": 7,
    "concat_ba_del_o": 8,
}
REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


class MergeDataset(Dataset):
    """Custom PyTorch Dataset for the merge conflict data."""
    def __init__(self, file_path, limit=None):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.samples = json.load(f)
        if limit:
            self.samples = self.samples[:limit]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "a_o_ids": torch.tensor(sample['a_o_aligned'], dtype=torch.long),
            "o_a_ids": torch.tensor(sample['o_a_aligned'], dtype=torch.long),
            "b_o_ids": torch.tensor(sample['b_o_aligned'], dtype=torch.long),
            "o_b_ids": torch.tensor(sample['o_b_aligned'], dtype=torch.long),
            "labels": torch.tensor(sample['numeric_label'], dtype=torch.long),
            "descriptive_label": sample['descriptive_label']
        }

def compute_metrics(preds, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
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
    all_numeric_labels = []
    all_descriptive_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            a_o_ids = batch['a_o_ids'].to(device)
            o_a_ids = batch['o_a_ids'].to(device)
            b_o_ids = batch['b_o_ids'].to(device)
            o_b_ids = batch['o_b_ids'].to(device)
            labels = batch['labels'].to(device)
            descriptive_labels = batch['descriptive_label']

            outputs = model(a_o_ids, o_a_ids, b_o_ids, o_b_ids, labels=labels)
            loss = outputs['loss']
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                loss = loss.mean()
            logits = outputs['logits']

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_numeric_labels.extend(labels.cpu().numpy())
            all_descriptive_labels.extend(descriptive_labels)

    avg_loss = total_loss / len(data_loader)
    
    # --- Detailed Analysis ---
    results_by_label = defaultdict(lambda: {"true": [], "pred": []})
    for true_label, pred_label, desc_label in zip(all_numeric_labels, all_preds, all_descriptive_labels):
        results_by_label[desc_label]["true"].append(true_label)
        results_by_label[desc_label]["pred"].append(pred_label)

    logging.info("\n" + "="*20 + " DETAILED EVALUATION PER CATEGORY " + "="*20 + "\n")

    for desc_label, data in sorted(results_by_label.items()):
        y_true = data["true"]
        y_pred = data["pred"]
        
        if not y_true:
            continue

        logging.info(f"--- Classification Report for Category: '{desc_label}' (count: {len(y_true)}) ---")
        
        all_numeric_labels_in_group = sorted(list(set(y_true) | set(y_pred)))
        target_names = [REVERSE_LABEL_MAP.get(l, f"unknown_label_{l}") for l in all_numeric_labels_in_group]

        try:
            report = classification_report(
                y_true, 
                y_pred, 
                labels=all_numeric_labels_in_group,
                target_names=target_names,
                zero_division=0
            )
            logging.info("\n" + report) # Add newline for better formatting
        except Exception as e:
            logging.warning(f"Could not generate classification report for '{desc_label}': {e}")
            acc = accuracy_score(y_true, y_pred)
            logging.info(f"  - Overall Accuracy for this group: {acc:.4f}")
        
        logging.info("--------------------------------------------------------------------\n")

    # Overall metrics
    metrics = compute_metrics(all_preds, all_numeric_labels)
    return avg_loss, metrics

def main():
    """Main function to run the training."""
    log_filename = f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info("\n\n" + "="*50 + " NEW TRAINING RUN " + "="*50 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    dataset_name = "mergebert_all_lang_with_no_newline"
    train_file = os.path.join(DATA_DIR, f"{dataset_name}_train.json")
    val_file = os.path.join(DATA_DIR, f"{dataset_name}_validation.json")

    train_dataset = MergeDataset(train_file)
    val_dataset = MergeDataset(val_file)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = MergeBERTModel(encoder_path=MODEL_PATH, num_labels=NUM_LABELS)
    model.to(device)

    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_val_accuracy = 0

    for epoch in range(EPOCHS):
        logging.info(f"Epoch {epoch + 1}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        logging.info(f"Train loss: {train_loss:.4f}")

        val_loss, metrics = eval_epoch(model, val_loader, device)
        logging.info(f"Validation loss: {val_loss:.4f}")
        logging.info(f"Overall validation metrics: {metrics}")

        if metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = metrics['accuracy']
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), os.path.join(SAVED_MODEL_DIR, 'mergebert.pt'))
            logging.info(f"Best model saved to {SAVED_MODEL_DIR}")

if __name__ == "__main__":
    main()