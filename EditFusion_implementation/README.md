
# EditFusion Implementation

---

## Overview

This repository provides a framework for training, evaluating, and serving models that infer edit scripts for code merge scenarios. It includes dataset collection, model training, and a Flask-based API for inference.

## Directory Structure

- `flask_service/`: REST API service for model inference.
  - `flask_app.py`: Main Flask application.
  - `http_test.py`: Example/test script for API requests.
- `train_and_infer/`: Model training, evaluation, and utilities.
  - `model/`: Model architecture definitions.
  - `utils/`: Utility functions (edit script generation, conflict handling, tokenization, etc.).
  - `collect_dataset.py`: Script for dataset creation and preprocessing.
  - `train.py`: Model training entry point.
  - `chunk_level_test.py`: Model evaluation/testing script.
  - `infer.py`: Inference logic for serving predictions.

## Installation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Pretrained Model Setup

To use the CodeBERTa model, download the pretrained weights to `train_and_infer/bert`:

```bash
git lfs install
git clone https://huggingface.co/huggingface/CodeBERTa-small-v1 train_and_infer/bert/CodeBERTa-small-v1
```

## Usage

### 1. Dataset Collection

1. Use [gitMergeScenario](https://github.com/Cltsu/gitMergeScenario) to collect conflict folders from repositories.
2. Aggregate all conflict blocks into a single JSON file:
   ```bash
   python -m train_and_infer.gather_conflict_data
   ```
3. Preprocess and collect the dataset:
   ```bash
   python -m train_and_infer.collect_dataset
   ```

#### Data Format Example

```json
[
  {
    "ours": ["line1", "line2", ...],
    "theirs": ["line1", "line2", ...],
    "base": ["line1", "line2", ...],
    "resolve": ["line1", "line2", ...]
  },
  ...
]
```

### 2. Model Training

Edit dataset/model paths in `train.py` as needed, then run:

```bash
python -m train_and_infer.train
```

### 3. Model Evaluation

Edit dataset/model paths in `chunk_level_test.py` as needed, then run:

```bash
python -m train_and_infer.chunk_level_test
```

### 4. Start Flask Service

Ensure the model path in `train_and_infer/infer.py` is correct, then start the API:

```bash
python -m flask_service.flask_app
```
Default port: 5002. Endpoint: `/es_predict` (GET/POST, params: `ours`, `theirs`, `base`).

### 5. Test Flask Service

Edit test data in `flask_service/http_test.py` as needed, then run:

```bash
python -m flask_service.http_test
```

---

For questions, issues, or contributions, please open an issue or pull request.