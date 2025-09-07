# MergeBERT MVP Implementation

This directory contains a simplified MVP (Minimum Viable Product) implementation of the MergeBERT model, based on the paper "Program Merge Conflict Resolution via Neural Transformers."

## Project Structure

- `CodeBERTa-small-v1/`: The pretrained CodeBERT model used as the backbone.
- `data/`: Will contain processed data for training.
- `saved_models/`: Will contain fine-tuned model checkpoints.
- `utils.py`: Helper functions for data processing, including `diff3` and sequence alignment.
- `preprocess.py`: Script to preprocess the raw data into a format suitable for training.
- `model.py`: The PyTorch implementation of the MergeBERT model architecture.
- `train.py`: The script for fine-tuning the MergeBERT model.
- `inference.py`: A script to run inference with a trained model on a new conflict.
- `requirements.txt`: Python dependencies.

## Usage

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Preprocess data:**
    ```bash
    python preprocess.py
    ```
3.  **Train the model:**
    ```bash
    python train.py
    ```
4.  **Run inference:**
    ```bash
    python inference.py --file_a <path_to_A> --file_b <path_to_B> --file_o <path_to_O>
    ```
