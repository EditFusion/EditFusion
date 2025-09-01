# EditFusion: A Deep Learning Approach to Merge Conflict Resolution

This project implements a model that predicts how to resolve three-way merge conflicts by composing edit scripts from the conflicting `ours` and `theirs` branches.

***

## Directory Structure

-   `/flask_service`: A Flask-based web service for inference.
    -   `flask_app.py`: The main Flask application that exposes the prediction API.
    -   `http_test.py`: A script to test the Flask service.
-   `/train_and_infer`: Contains scripts and modules for model training, evaluation, and inference.
    -   `/model`: Defines the neural network architectures (e.g., LSTM, GRU) and embedding layers.
    -   `/trainers`: Contains the training and evaluation logic for the models.
    -   `/utils`: Includes utility functions for handling conflicts, generating edit scripts, and tokenization.
    -   `collect_dataset.py`: Script to process raw conflict data into a trainable dataset.
    -   `gather_conflict_data.py`: Script to aggregate conflict data from multiple sources.
    -   `train.py`: The main script for training the model.
    -   `chunk_level_test.py`: Script to evaluate the trained model on a test dataset.
    -   `infer.py`: Provides the core inference function used by the Flask service.

## Setup

### 1. Install Dependencies

Install the required Python packages using `requirements.txt`.

```bash
pip install -r requirements.txt
```

### 2. Download Pre-trained Model

The model uses a pre-trained language model for code (e.g., CodeBERTa, GraphCodeBERT). You need to download it into the `train_and_infer/bert` directory.

```bash
# Set working directory to EditFusion_implementation/train_and_infer/bert
git lfs install
git clone https://huggingface.co/huggingface/CodeBERTa-small-v1
```

## Usage Workflow

The process involves collecting data, training a model, and deploying it for inference.

### 1. Data Collection

Data collection is a three-stage process to create a dataset suitable for training.

**Stage 1: Collect Raw Conflict Scenarios**
First, use the tool in the `../gitMergeScenario` directory to mine git repositories for historical merge conflicts. This stage outputs a directory for each conflict, containing the `ours`, `theirs`, `base`, and resolved versions of the file.

**Stage 2: Consolidate Conflict Data**
Next, aggregate all the collected conflict chunks from the file system into a single JSON file. This script also assigns a preliminary `resolution_kind` label to each conflict.

```bash
# Set working directory to the project root (EditFusion_implementation)
python -m train_and_infer.gather_conflict_data
```

**Stage 3: Generate Trainable Dataset**
Finally, process the consolidated JSON file to generate the final dataset in CSV format. This script computes edit scripts for each conflict, filters for resolvable conflicts using a backtracking algorithm, and tokenizes the data. This process is parallelized to speed up collection.

Before running, modify `collect_dataset.py` to set the input and output file paths.

```bash
# Set working directory to the project root (EditFusion_implementation)
python -m train_and_infer.collect_dataset
```

### 2. Model Training

Train the model using the generated CSV dataset. Before running, you may need to adjust the dataset path and output model name inside the `train.py` script.

The training process can be accelerated using `accelerate` for multi-GPU setups.

```bash
# Set working directory to the project root (EditFusion_implementation)

# Configure accelerate for your environment
accelerate config

# Launch training
accelerate launch -m train_and_infer.train
```

### 3. Model Evaluation

Evaluate the performance of the trained model on the test set. Modify `chunk_level_test.py` to specify the dataset and the trained model path.

```bash
# Set working directory to the project root (EditFusion_implementation)
python -m train_and_infer.chunk_level_test
```

### 4. Start Inference Service

To use the model for predictions, start the Flask service. Ensure the model path in `train_and_infer/infer.py` points to your trained model.

The service runs on port 5002 by default and provides the `/es_predict` endpoint.

```bash
# Set working directory to the project root (EditFusion_implementation)
python -m flask_service.flask_app
```

The API accepts GET requests with three string parameters: `ours`, `theirs`, and `base`.

### 5. Test the Service

Use the `http_test.py` script to send a sample request to the running Flask service and verify its output. You can modify the test data within the script.

```bash
# Set working directory to the project root (EditFusion_implementation)
python -m flask_service.http_test
```
