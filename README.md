# Optimal Privacy Budget Allocation Framework for Medical Data Publishing

This repository provides the implementation of the experiments in our paper: "Optimal Privacy Budget Allocation Framework for Medical Data Publishing". The code reproduces the main results, statistical tests, and figures reported in the paper.

## Repository Structure

```
├── dataset/                                     # Dataset used in experiments
├── fig_out/                                     # Output figures and plots
├── 1.OPBA-MDP_v1.5.py                           # Main algorithm implementation (training & evaluation)
├── 2.Significance_test.py                       # Statistical significance testing (e.g., Wilcoxon test)
├── 3.Figure_metrics_vs_budgets_per_dataset_1.1.py # Generate performance vs. budget figures

```

## Requirements

* Python >= 3.9

* Dependencies:

  ```
  pip install torch numpy matplotlib scipy
  ```

* Standard library modules used:
  os, random, time, csv, math, statistics

## Usage

### Run the main algorithm

Configure the experiment settings (e.g., dataset path, privacy budgets, random seeds) before running:

```
python 1.OPBA-MDP_v1.5.py
```

This script executes the proposed privacy budget allocation process and saves results automatically in result/.

### Run statistical significance tests

To evaluate the robustness of experimental results across datasets:

```
python 2.Significance_test.py
```

This script performs pairwise Wilcoxon signed-rank tests and outputs tabular results for reporting.

### Generate figures

* Metrics vs. Budgets across datasets:

  ```
  python 3.Figure_metrics_vs_budgets_per_dataset_1.1.py
  ```

All figures will be saved to the folder: fig_out/
(Each figure can be directly used in paper visualization.)

## Dataset

The dataset/ folder contains sample datasets used in our experiments.
The main experiments use standard benchmark datasets, including:

* Heart Disease Dataset (UCI Machine Learning Repository) https://archive.ics.uci.edu/dataset/45/heart+disease

* Diabetes Classification Dataset (Kaggle) https://www.kaggle.com/datasets/simaanjali/diabetes-classification-dataset

## License

This repository is released under the MIT License.
