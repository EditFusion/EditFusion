from tqdm import tqdm
import time
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pickle
import os
import random
import torch
import json
from pathlib import Path
from torch.utils.data import DataLoader, random_split

# Use relative imports as this will be run as a module
from .model.LSTM_model import LSTMClassifier
from .model.CCEmbedding.MergeBertCCEmbedding import MergeBertCCEmbedding
from .utils.util import ObjectDict

def seed_everything(seed=42):
    """
    Set a seed for reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def mask_output(outputs, labels, lengths):
    """
    Masks the output and labels based on the actual sequence lengths.
    """
    device = outputs.device
    batch_size, seq_len, _ = outputs.size()
    max_length = lengths.max().item()
    max_length = min(max_length, seq_len)

    outputs = outputs[:, :max_length, :].squeeze(2)
    labels = labels[:, :max_length]

    mask = torch.arange(max_length, device=device).expand(
        batch_size, max_length
    ) < lengths.unsqueeze(1)

    outputs_selected = outputs.masked_select(mask).view(-1)
    labels_selected = labels.masked_select(mask).view(-1)
    return outputs_selected, labels_selected

if __name__ == "__main__":
    seed_everything()

    # Use the model directory provided by the user
    model_dir = Path("/home/foril/projects/EditFusion/EditFusion_implementation/train_and_infer/data/model_output/lstm_codebert_mergebert_all_lang_08-31-21:45:11")
    model_param_path = model_dir / "best_model.pth"

    # Find parameter files automatically
    try:
        model_params_path = next(model_dir.glob("model_params_*.json"))
        training_param_path = next(model_dir.glob("training_params_*.json"))
    except StopIteration:
        print(f"Error: Could not find parameter files in {model_dir}")
        exit()

    with open(training_param_path, 'r') as f:
        training_param_dict = json.load(f)
        dataset_name = "codebert_mergebert_all_lang"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    with open(model_params_path, 'r') as f:
        model_params = json.load(f)

    training_param = ObjectDict(training_param_dict)

    batch_size = 16

    script_path = Path(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = script_path / "data" / "processed_data" / dataset_name

    print('Initializing model...')
    model = LSTMClassifier(**model_params, CCEmbedding_class=MergeBertCCEmbedding)
    
    state_dict = torch.load(model_param_path, map_location=device)
    if next(iter(state_dict)).startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    embedding_class = MergeBertCCEmbedding()
    full_dataset = embedding_class.get_dataset(dataset_path)
    print(f"Dataset size: {len(full_dataset)}")

    total_size = len(full_dataset)
    train_size = int(total_size * training_param.TRAIN_SPLIT)
    val_size = int(total_size * training_param.VAL_SPLIT)
    test_size = total_size - train_size - val_size
    print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=embedding_class.collate_fn,
    )

    all_labels = []
    all_probs = []

    start_time = time.perf_counter()
    with torch.no_grad():
        for loaded_feats, labels, lengths, resolution_kinds in tqdm(test_loader, desc="Evaluating on test set"):
            labels = labels.to(device)
            lengths = lengths.to(device)
            if isinstance(loaded_feats, tuple):
                loaded_feats = tuple(f.to(device) for f in loaded_feats)

            outputs = model(loaded_feats, lengths)
            
            outputs_selected, labels_selected = mask_output(outputs, labels, lengths)
            
            probabilities = torch.sigmoid(outputs_selected)
            
            all_labels.append(labels_selected.cpu().numpy())
            all_probs.append(probabilities.cpu().numpy())

    end_time = time.perf_counter()
    print(f"Evaluation finished in { (end_time - start_time) * 1000:.2f} ms")

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    model_name = model_dir.name
    save_dir = script_path / "roc_data"
    os.makedirs(save_dir, exist_ok=True)
    
    save_name = "MergeBert_model_for_ROC.pkl"
    save_path = save_dir / save_name

    with open(save_path, 'wb') as f:
        pickle.dump({
            'labels': all_labels,
            'probabilities': all_probs,
            'model_name': "EditFusion (MergeBert)"
        }, f)

    print(f"ROC data saved to {save_path}")

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for EditFusion (MergeBert)')
    plt.legend(loc="lower right")

    plot_save_path = save_dir / 'roc_curve_mergebert.png'
    plt.savefig(plot_save_path)
    print(f"ROC curve plot saved as '{plot_save_path}' with AUC: {roc_auc:.4f}")
