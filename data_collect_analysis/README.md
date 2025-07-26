
# Data Collection and Analysis Scripts

This folder contains scripts and resources for collecting, analyzing, and evaluating code merge algorithms. It is designed for research and experimentation with different diff and merge strategies, primarily on datasets from MergeBert (Java, C#, JavaScript, TypeScript).

## Folder Structure and Purpose

- `bt_expe.py`: Main script for running experiments and evaluations on merge algorithms.
- `data_collect_analysis/`: Contains additional scripts for data collection and merging, including:
  - `merge-file-collect100+stars.py`: Script for collecting files from repositories with 100+ stars.
  - `script.ipynb`: Jupyter notebook for running and visualizing algorithm results on the MergeBert dataset.

## Usage Instructions

1. **Run Experiments**
	- Use `bt_expe.py` to execute merge algorithm experiments. Configure the script as needed for your dataset and algorithm.

2. **Data Collection**
	- Use `merge-file-collect100+stars.py` to gather source files from popular repositories for analysis.

3. **Algorithm Evaluation**
	- Open `script.ipynb` in Jupyter Notebook to view and analyze the results of merge algorithms on the MergeBert dataset.

## Additional Information

- The folder is intended for algorithm research and evaluation. Scripts are modular and can be adapted for different datasets or merge strategies.
- For theoretical background and custom Git source code, see the [Git/expe_foril branch](https://github.com/foriLLL/git/tree/expe_foril). After compiling the Git executable, specify its path in `git_replay.py` to run experiments. Results will be saved in `output/result_*.txt`.

Feel free to use, modify, and extend these scripts for your own research or projects. For questions or contributions, please open an issue or pull request.
