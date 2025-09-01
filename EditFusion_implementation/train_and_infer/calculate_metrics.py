import pickle
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
import argparse

def calculate_metrics_from_pkl(file_path, threshold=0.5):
    """
    Loads ROC data from a .pkl file and calculates precision, recall, and F1-score.
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    labels = data['labels']
    probs = data['probabilities']
    model_name = data.get('model_name', Path(file_path).stem)

    # Apply threshold to get binary predictions
    predictions = (probs >= threshold).astype(int)

    # Calculate metrics
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    
    # For more detail, let's get the confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    print(f"\nMetrics for model: '{model_name}' (at threshold={threshold})")
    print(f"--------------------------------------------------")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"--------------------------------------------------")
    print("Confusion Matrix:")
    print(f"  - True Positives (TP):  {tp}")
    print(f"  - False Positives (FP): {fp}")
    print(f"  - True Negatives (TN):  {tn}")
    print(f"  - False Negatives (FN): {fn}")
    print(f"--------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate precision, recall, and other metrics from a .pkl file.')
    parser.add_argument('file', nargs='?', help='Path to the .pkl file containing ROC data.')
    args = parser.parse_args()

    if args.file:
        target_pkl_file = Path(args.file)
        if target_pkl_file.exists():
            calculate_metrics_from_pkl(target_pkl_file)
        else:
            print(f"Error: The specified file does not exist: {target_pkl_file}")
    else:
        # If no file is specified, find the one for the MergeBert model automatically
        script_path = Path(__file__).parent
        roc_data_dir = script_path / "roc_data"
        
        target_pkl_file = roc_data_dir / "seperate edit type embedding.pkl"

        if target_pkl_file.exists():
            calculate_metrics_from_pkl(target_pkl_file)
        else:
            # Fallback to the original name if the new one is not found
            fallback_file = roc_data_dir / "MergeBert_model_for_ROC.pkl"
            if fallback_file.exists():
                calculate_metrics_from_pkl(fallback_file)
            else:
                print(f"Error: Could not find the .pkl file for the MergeBert model in {roc_data_dir}")
