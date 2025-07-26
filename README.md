# EditFusion Project

This repository provides a comprehensive framework for mining, collecting, analyzing, and modeling code merge scenarios, with a focus on conflict resolution and edit script inference. The project integrates data mining, dataset creation, model training/evaluation, and API serving, supporting both research and practical applications.

## Directory Overview

- **EditFusion_implementation/**
  - Main implementation for model training, evaluation, and inference.
  - Includes a Flask-based REST API for serving predictions.
  - Contains scripts for dataset collection, model definition, training, evaluation, and inference.
  - See [EditFusion_implementation/README.md](./EditFusion_implementation/README.md) for detailed usage and setup.

- **gitMergeScenario/**
  - Java-based tools for mining and collecting conflict files and tuples from Git repositories.
  - Supports automated and manual repository analysis, conflict tuple extraction, and dataset generation.
  - See [gitMergeScenario/README.md](./gitMergeScenario/README.md) for setup, input/output formats, and troubleshooting.

- **GraphQL_repo_mining/**
  - Python scripts and notebooks for mining GitHub repositories using the GraphQL API.
  - Enables filtering, analysis, and extraction of repository metadata based on custom criteria.
  - See [GraphQL_repo_mining/README.md](./GraphQL_repo_mining/README.md) for mining instructions and requirements.

- **data_collect_analysis/**
  - Scripts and notebooks for collecting, analyzing, and evaluating code merge algorithms and datasets (e.g., MergeBert).
  - Supports experiment automation, data collection from popular repositories, and algorithm evaluation.
  - See [data_collect_analysis/README.md](./data_collect_analysis/README.md) for experiment setup and usage.

## Getting Started

1. **Mining and Collecting Data**
   - Use `gitMergeScenario` to collect conflict files and tuples from repositories.
   - Use `GraphQL_repo_mining` to mine repository metadata from GitHub.
   - Use `data_collect_analysis` for additional data collection and algorithm evaluation.

2. **Dataset Preparation**
   - Aggregate and preprocess conflict data using scripts in `EditFusion_implementation/train_and_infer/`.
   - See the respective README files for data format and preprocessing steps.

3. **Model Training and Evaluation**
   - Train and evaluate models using scripts in `EditFusion_implementation/train_and_infer/`.
   - Adjust dataset/model paths as needed.
   - See [EditFusion_implementation/README.md](./EditFusion_implementation/README.md) for details.

4. **API Service**
   - Serve model predictions via the Flask API in `EditFusion_implementation/flask_service/`.
   - See [EditFusion_implementation/README.md](./EditFusion_implementation/README.md) for API usage and endpoints.

## Requirements
- Python 3.x (for Python scripts)
- Java (for gitMergeScenario)
- See each subdirectory's README for specific package and environment requirements.

## Notes
- Each subdirectory contains a dedicated README with detailed instructions, usage examples, and troubleshooting tips.
- For advanced usage, dataset formats, and experiment configurations, refer to the README in the relevant directory:
  - [EditFusion_implementation/README.md](./EditFusion_implementation/README.md)
  - [gitMergeScenario/README.md](./gitMergeScenario/README.md)
  - [GraphQL_repo_mining/README.md](./GraphQL_repo_mining/README.md)
  - [data_collect_analysis/README.md](./data_collect_analysis/README.md)

For questions, issues, or contributions, please open an issue or pull request in this repository.
