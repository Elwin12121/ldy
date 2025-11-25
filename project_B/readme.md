CIFAR-10 Imbalanced Classification with Two-Stage Dynamic Reweighting (ICTTA Framework)
1. Overview

This project implements a Two-Stage Dynamic Reweighting Framework (Two-Stage ICTTA) for imbalanced CIFAR-10 classification, combining ideas from:

Decoupled Representation Learning for Long-Tailed Recognition (Kang et al., 2020)

ICTTA: Inter-Class and Temporal Transfer Attention (Zhou et al., 2023)

The goal is to verify the effectiveness of a two-stage adaptive weighting strategy under both long-tail imbalance and head-tail imbalance scenarios.

2. Project Structure
Project/
│
├── data_loader.py             # CIFAR-10 loader with long-tail / head-tail imbalance setup
├── data_loading.py            # Legacy data loading utilities
├── model.py                   # Model definitions: SimpleCNN / ResNet10 / ResNet15 / ResNet20
├── losses.py                  # Focal loss and dynamic weight computation functions
├── train_eval.py              # Training & evaluation logic (train_one_epoch / evaluate)
├── results_logger.py          # Save training metrics (loss, acc, F1) per epoch to CSV
│
├── two_stage_ictta_v3.py      # Core Two-Stage ICTTA training framework
├── baseline_experiments.py    # Early baseline experiment runner
├── ictta_tuning_resnet10.py   # Tuning script for ResNet10
├── ictta_tuning_resnet15_v2.py# Tuning script for ResNet15
├── ictta_tuning_resnet20_v2.py# Tuning script for ResNet20
│
├── run_all_models.py          # Run all baseline & tuning experiments
├── summary_tunning.py         # Summarize results from multiple tuning experiments
├── plots.py                   # Visualization pipeline (bar, line, heatmap, training curves)
├── main.py                    # Optional entry point for single runs
└── figures/                   # Output folder (auto-generated after running)

3. Environment Requirements

Ensure the following dependencies are installed:

Python >= 3.8
torch >= 2.0
torchvision
scikit-learn
matplotlib
pandas
numpy


Install them via:

pip install torch torchvision scikit-learn matplotlib pandas numpy

4. How to Run
1). Run All Baseline Experiments

Run all four models (SimpleCNN, ResNet10, ResNet15, ResNet20) on both imbalance scenarios:

python run_all_models.py


After running, result logs will be generated automatically:

results_baseline_resnet10_long_tail.csv
results_baseline_resnet15_head_tail.csv
...

2). Small-Range Hyperparameter Tuning

run_all_models.py also performs small-range tuning for ResNet10/15/20:

python run_all_models.py


It automatically calls:

run_baseline_all_models() for baseline results

run_tuning_resnets() for tuning different learning rates, weight ranges, and focal gamma values.

3). Run a Single Model Manually

Example for running ResNet15 on the long-tail scenario:

from two_stage_ictta_v3 import run_two_stage_ictta_on_scenario

run_two_stage_ictta_on_scenario(
    model_name="resnet15",
    imbalance_type="long_tail",
    stage1_epochs=20,
    stage2_epochs=20,
    imb_factor=0.01,
    batch_size=128,
    log_dir=".",
    stage1_lr=0.1,
    stage2_lr=0.01,
    min_weight=1.0,
    max_weight=8.0,
    ema_alpha=0.8,
    stage2_loss_type="focal",
    stage2_focal_gamma=1.5,
    dynamic_mode="loss",
    exp_name="custom_run_resnet15"
)

4). Summarize Tuning Results

Generate a summary file of all tuning runs:

python summary_tunning.py


This will create summary_tuning.csv.

5). Visualization Pipeline

Generate all figures and tables automatically:

python plots.py


The script will produce grouped bar charts, line charts, heatmaps, and best-run training curves.

Example outputs:

figures/
│── bar_models_vs_scenario_macro_f1.png
│── line_depth_trend_macro_f1.png
│── heatmap_macro_f1.png
│── best_summary.csv
│── best_hparams.csv
│── curves_best/
│   ├── resnet10_long_tail_acc_curves_best.png
│   └── resnet15_head_tail_acc_curves_best.png
└── class_counts_imb01.csv

5. Module Descriptions
Module	Description
two_stage_ictta_v3.py	Implements Stage 1 (representation learning) + Stage 2 (dynamic adaptation).
train_eval.py	Defines core training and evaluation functions with CE/Focal Loss support.
losses.py	Contains FocalLoss class and per-class dynamic weight computations.
results_logger.py	Logs per-epoch results (loss, accuracy, F1 metrics) to CSV.
plots.py	Generates grouped bar charts, line trends, accuracy curves, and heatmaps.
run_all_models.py	High-level runner for all model-scenario combinations.
summary_tunning.py	Aggregates multiple experiment logs into one summary file.

6. Log File Format

Each results_*.csv contains the following columns:

Column	Description
stage	Stage name (e.g., stage1_long_tail / stage2_head_tail)
epoch	Epoch number
model	Model name
method	Training method (e.g., two_stage_ictta_v3)
loss	Average training loss
acc	Overall accuracy
macro_f1	Macro-average F1 score
tail_macro_f1	Macro F1 over tail classes

7. Recommended Analysis

Focus on:

Stage1 vs Stage2 improvement → verifies the effectiveness of two-stage training

Tail Macro-F1 increase → measures enhancement for minority classes

Depth trend → e.g., ResNet15 often balances feature capacity and stability best

8. Notes

Default imbalance factor: imb_factor = 0.01 (tail classes = 1% of head samples)

Use GPU if available (cuda automatically detected).

Start with small models (e.g., ResNet10) to validate the pipeline before full runs.
