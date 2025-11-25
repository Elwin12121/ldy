# plots.py
# ---------------------------------------------------------
# Visualization utilities for CIFAR-10 long-tail experiments.
#
# This module assumes that:
#   - Training logs are saved as CSV files named:
#       results_<exp_name>.csv
#   - Each CSV has columns:
#       stage, epoch, model, method, loss, acc,
#       macro_f1, tail_macro_f1
#
# It provides:
#   1) Loading all result files
#   2) Summarizing the best epoch per (model, scenario)
#   3) Plotting:
#       - Grouped bar charts (models × scenarios)
#       - Performance vs model depth line chart
#       - Heatmap for model × scenario × metric
#       - Accuracy curves for the best run per (model, scenario)
#   4) Tables:
#       - Best summary table
#       - Hyper-parameter table for best runs
#       - Class count table for long_tail and head_tail scenarios
# ---------------------------------------------------------

import glob
import os
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------
# 1. Loading and summarization
# ---------------------------------------------------------

def load_all_results(results_dir: str = ".",
                     pattern: str = "results_*.csv") -> pd.DataFrame:
    """
    Load all CSV result logs in a directory and concatenate them.

    Each CSV is expected to have:
        stage, epoch, model, method,
        loss, acc, macro_f1, tail_macro_f1

    New columns:
        - source_file: the file stem (without extension)
        - exp        : experiment name derived from file name
    """
    search_pattern = os.path.join(results_dir, pattern)
    csv_paths = glob.glob(search_pattern)

    if not csv_paths:
        raise FileNotFoundError(f"No CSV files matched: {search_pattern}")

    all_dfs: List[pd.DataFrame] = []
    for path in csv_paths:
        df = pd.read_csv(path)
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]  # e.g. results_resnet10_head_tail_r10_w1-8_lr2
        df["source_file"] = stem

        # Derive experiment name by stripping "results_" prefix if present
        exp_name = stem[8:] if stem.startswith("results_") else stem
        df["exp"] = exp_name

        all_dfs.append(df)

    all_results = pd.concat(all_dfs, ignore_index=True)
    return all_results


def _parse_scenario_from_strings(exp: str, stage: str, source_file: str) -> str:
    """
    Infer scenario name ("head_tail", "long_tail", or "unknown")
    from exp name / stage / source_file string.
    """
    text = f"{exp} {stage} {source_file}".lower()
    if "head_tail" in text or "headtail" in text:
        return "head_tail"
    if "long_tail" in text or "longtail" in text:
        return "long_tail"
    return "unknown"


def add_scenario_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'scenario' column to the given DataFrame using information
    from 'exp', 'stage' and 'source_file'.
    """
    df = df.copy()
    if "exp" not in df.columns:
        df["exp"] = ""

    if "source_file" not in df.columns:
        df["source_file"] = ""

    df["scenario"] = df.apply(
        lambda row: _parse_scenario_from_strings(
            str(row.get("exp", "")),
            str(row.get("stage", "")),
            str(row.get("source_file", "")),
        ),
        axis=1,
    )
    return df


def summarize_best_by_model_and_scenario(all_results: pd.DataFrame,
                                         metric: str = "macro_f1") -> pd.DataFrame:
    """
    For each (model, scenario), pick the epoch with the best given metric.

    Args:
        all_results: concatenated DataFrame of all experiments.
        metric     : one of 'acc', 'macro_f1', 'tail_macro_f1'.

    Returns:
        summary_df with columns:
            model, scenario, epoch, stage,
            acc, macro_f1, tail_macro_f1,
            exp, source_file
    """
    if metric not in ["acc", "macro_f1", "tail_macro_f1"]:
        raise ValueError(f"Unsupported metric: {metric}")

    df = add_scenario_column(all_results)
    df = df[df["scenario"].isin(["head_tail", "long_tail"])].reset_index(drop=True)

    group_cols = ["model", "scenario"]
    idx = df.groupby(group_cols)[metric].idxmax()

    best_df = (
        df.loc[idx, [
            "model", "scenario", "epoch", "stage",
            "acc", "macro_f1", "tail_macro_f1",
            "exp", "source_file",
        ]]
        .sort_values(["model", "scenario"])
        .reset_index(drop=True)
    )

    return best_df


# ---------------------------------------------------------
# 2. Basic plotting helpers
# ---------------------------------------------------------

def _ensure_output_dir(save_path: str):
    """
    Ensure that the directory for the given path exists.
    """
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def plot_grouped_bar(summary_df: pd.DataFrame,
                     metric: str = "macro_f1",
                     save_path: str = "figures/bar_models_vs_scenario_macro_f1.png"):
    """
    Plot a grouped bar chart:

        x-axis: model
        bars  : head_tail vs long_tail
        y-axis: selected metric

    Args:
        summary_df: DataFrame from summarize_best_by_model_and_scenario().
        metric    : 'acc', 'macro_f1', or 'tail_macro_f1'.
        save_path : path to save the figure.
    """
    _ensure_output_dir(save_path)

    if metric not in ["acc", "macro_f1", "tail_macro_f1"]:
        raise ValueError(f"Unsupported metric: {metric}")

    # Human-readable labels for metrics (used in titles / axes)
    metric_label_map = {
        "acc": "Accuracy",
        "macro_f1": "Macro-F1",
        "tail_macro_f1": "Tail Macro-F1",
    }
    metric_label = metric_label_map.get(metric, metric)

    df = summary_df.copy()
    df["model"] = df["model"].str.lower()

    # Exclude SimpleCNN from all grouped bar plots
    model_order = ["resnet10", "resnet15", "resnet20"]
    df = df[df["model"].isin(model_order)]

    pivot = df.pivot(index="model", columns="scenario", values=metric)
    pivot = pivot.reindex(model_order)

    ax = pivot.plot(kind="bar", figsize=(8, 5))
    ax.set_xlabel("Model")
    ax.set_ylabel(metric_label)
    ax.set_title(f"Best {metric_label} across models and scenarios")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=0)

    # Fix the y-axis range for consistency across all plots
    ax.set_ylim(0, 1.0)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_depth_trend(summary_df: pd.DataFrame,
                     metric: str = "macro_f1",
                     save_path: str = "figures/line_depth_trend_macro_f1.png"):
    """
    Plot a line chart to show performance vs model depth:

        ResNet10 -> ResNet15 -> ResNet20

    Each scenario ('head_tail', 'long_tail') becomes one curve.
    """
    _ensure_output_dir(save_path)

    if metric not in ["acc", "macro_f1", "tail_macro_f1"]:
        raise ValueError(f"Unsupported metric: {metric}")

    metric_label_map = {
        "acc": "Accuracy",
        "macro_f1": "Macro-F1",
        "tail_macro_f1": "Tail Macro-F1",
    }
    metric_label = metric_label_map.get(metric, metric)

    # Depth mapping excludes SimpleCNN
    depth_map = {
        "resnet10": 10,
        "resnet15": 15,
        "resnet20": 20,
    }

    df = summary_df.copy()
    df["model"] = df["model"].str.lower()
    df = df[df["model"].isin(depth_map.keys())]
    df["depth"] = df["model"].map(depth_map)

    plt.figure(figsize=(8, 5))

    for scenario in ["head_tail", "long_tail"]:
        sub = df[df["scenario"] == scenario].sort_values("depth")
        if sub.empty:
            continue
        plt.plot(
            sub["depth"],
            sub[metric],
            marker="o",
            label=scenario,
        )

        for _, row in sub.iterrows():
            plt.text(
                row["depth"],
                row[metric] + 0.003,
                row["model"],
                ha="center",
                fontsize=8,
            )

    plt.xlabel("Model depth")
    plt.ylabel(metric_label)
    plt.title(f"{metric_label} vs model depth (best per scenario)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_training_curves_for_experiment(
    exp_name: str,
    results_dir: str = ".",
    save_prefix: str = "figures/curves",
):
    """
    Plot training curves (Stage 1 + Stage 2) for a single experiment.

    This function keeps Stage1 and Stage2 separate and uses different labels.
    For "best run" continuous curves, see plot_best_accuracy_curves_for_all.
    """
    csv_path = os.path.join(results_dir, f"results_{exp_name}.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Cannot find log file: {csv_path}")

    df = pd.read_csv(csv_path)

    def parse_stage_type(s: str) -> str:
        if isinstance(s, str) and s.startswith("stage1"):
            return "stage1"
        if isinstance(s, str) and s.startswith("stage2"):
            return "stage2"
        return s

    df["stage_type"] = df["stage"].astype(str).apply(parse_stage_type)

    metrics = ["acc", "macro_f1", "tail_macro_f1"]
    for metric in metrics:
        plt.figure(figsize=(8, 5))

        for stage_type in ["stage1", "stage2"]:
            sub = df[df["stage_type"] == stage_type].sort_values("epoch")
            if sub.empty:
                continue
            plt.plot(
                sub["epoch"],
                sub[metric],
                marker="o",
                label=stage_type,
            )

        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.title(f"{exp_name} – {metric} over epochs (Stage1 vs Stage2)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(save_prefix, f"{exp_name}_{metric}_curves.png")
        _ensure_output_dir(out_path)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------
# 3. Hyper-parameter table for best models
# ---------------------------------------------------------

def build_best_hparam_table(best_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Build a hyper-parameter table for the best (model, scenario) runs.

    This function decodes the experiment name `exp` and maps it to
    training hyper-parameters according to run_all_models.py:

        - Baseline:
            simplecnn:
                stage1_lr=0.05, stage2_lr=0.01,
                min_weight=1, max_weight=8, gamma=1.5
            resnet10/15/20:
                stage1_lr=0.1, stage2_lr=0.01,
                min_weight=1, max_weight=8, gamma=1.5

        - Tuning configs (ResNet10/15/20):
            r10_w1-5_lr2     -> (0.1, 0.01, 1, 5, 1.5)
            r10_w1-8_lr2     -> (0.1, 0.01, 1, 8, 1.5)
            r10_w2-8_lrsmall -> (0.1, 0.005, 2, 8, 1.5)

            r15_lrsmall_w1-5_g1.5 -> (0.05, 0.005, 1, 5, 1.5)
            r15_lrsmall_w1-6_g1.5 -> (0.05, 0.005, 1, 6, 1.5)
            r15_lrsmall_w1-4_g1.5 -> (0.05, 0.005, 1, 4, 1.5)

            r20_lrsmall_w1-6_g1.5 -> (0.1, 0.005, 1, 6, 1.5)
            r20_lrsmall_w1-4_g1.5 -> (0.1, 0.005, 1, 4, 1.5)
            r20_lrsmall_w2-8_g1.5 -> (0.1, 0.005, 2, 8, 1.5)

    Returns:
        DataFrame with columns:
            model, scenario, exp,
            stage1_lr, stage2_lr, min_weight, max_weight, focal_gamma,
            acc, macro_f1, tail_macro_f1
    """

    cfg_map = {
        # ResNet10
        "r10_w1-5_lr2":      (0.1, 0.01, 1, 5, 1.5),
        "r10_w1-8_lr2":      (0.1, 0.01, 1, 8, 1.5),
        "r10_w2-8_lrsmall":  (0.1, 0.005, 2, 8, 1.5),

        # ResNet15
        "r15_lrsmall_w1-5_g1.5": (0.05, 0.005, 1, 5, 1.5),
        "r15_lrsmall_w1-6_g1.5": (0.05, 0.005, 1, 6, 1.5),
        "r15_lrsmall_w1-4_g1.5": (0.05, 0.005, 1, 4, 1.5),

        # ResNet20
        "r20_lrsmall_w1-6_g1.5": (0.1, 0.005, 1, 6, 1.5),
        "r20_lrsmall_w1-4_g1.5": (0.1, 0.005, 1, 4, 1.5),
        "r20_lrsmall_w2-8_g1.5": (0.1, 0.005, 2, 8, 1.5),
    }

    rows = []
    for _, row in best_summary.iterrows():
        model = str(row["model"]).lower()
        scenario = row["scenario"]
        exp = row["exp"]

        stage1_lr = None
        stage2_lr = None
        min_w = None
        max_w = None
        gamma = None

        if exp.startswith("baseline_"):
            if model == "simplecnn":
                stage1_lr = 0.05
            else:
                stage1_lr = 0.1
            stage2_lr = 0.01
            min_w = 1.0
            max_w = 8.0
            gamma = 1.5
        else:
            # Example: resnet15_head_tail_r15_lrsmall_w1-5_g1.5
            parts = exp.split("_")
            cfg_name = parts[-1] if len(parts) > 0 else ""

            if cfg_name in cfg_map:
                stage1_lr, stage2_lr, min_w, max_w, gamma = cfg_map[cfg_name]

        rows.append({
            "model": model,
            "scenario": scenario,
            "exp": exp,
            "stage1_lr": stage1_lr,
            "stage2_lr": stage2_lr,
            "min_weight": min_w,
            "max_weight": max_w,
            "focal_gamma": gamma,
            "acc": row["acc"],
            "macro_f1": row["macro_f1"],
            "tail_macro_f1": row["tail_macro_f1"],
        })

    hparam_df = pd.DataFrame(rows)
    return hparam_df


def save_best_hparam_table(best_summary: pd.DataFrame,
                           out_path: str = "figures/best_hparams.csv") -> None:
    """
    Build hyper-parameter table and save to CSV.
    """
    directory = os.path.dirname(out_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    hparam_df = build_best_hparam_table(best_summary)
    hparam_df.to_csv(out_path, index=False)
    print(f"[plots] Best hyper-parameter table saved to: {out_path}")


# ---------------------------------------------------------
# 4. Accuracy curves for best runs (continuous Stage1+Stage2)
# ---------------------------------------------------------

def plot_best_accuracy_curves_for_all(
    best_summary: pd.DataFrame,
    results_dir: str = ".",
    out_dir: str = "figures/curves_best",
):
    """
    Plot two-stage accuracy curves for best runs:
        - Stage1: blue line
        - Stage2: orange line
        - Vertical dashed line at boundary
        - Orange segment connects the last point of Stage1 and the first of Stage2
    Titles use formal model and scenario names.

    SimpleCNN is skipped so that only ResNet models are visualized.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Mapping from model codes to display names
    model_name_map = {
        "simplecnn": "SimpleCNN",
        "resnet10": "ResNet-10",
        "resnet15": "ResNet-15",
        "resnet20": "ResNet-20",
    }

    # Mapping from scenario to formal display name
    scenario_name_map = {
        "long_tail": "Long-Tail Imbalance",
        "head_tail": "Head-Tail Imbalance",
    }

    for _, row in best_summary.iterrows():
        exp = row["exp"]
        model = str(row["model"]).lower()
        scenario = row["scenario"]

        # Skip SimpleCNN for accuracy curve plots
        if model == "simplecnn":
            continue

        csv_path = os.path.join(results_dir, f"results_{exp}.csv")
        if not os.path.isfile(csv_path):
            print(f"[plots] Warning: log file not found for exp={exp}")
            continue

        df = pd.read_csv(csv_path)

        # Parse stage type
        def parse_stage_type(s: str) -> str:
            if isinstance(s, str) and s.startswith("stage1"):
                return "stage1"
            if isinstance(s, str) and s.startswith("stage2"):
                return "stage2"
            return s

        df["stage_type"] = df["stage"].astype(str).apply(parse_stage_type)

        stage1 = df[df["stage_type"] == "stage1"].sort_values("epoch")
        stage2 = df[df["stage_type"] == "stage2"].sort_values("epoch")

        if stage1.empty and stage2.empty:
            print(f"[plots] Warning: no stage1/stage2 data for exp={exp}")
            continue

        plt.figure(figsize=(8, 5))

        # Plot Stage1 in blue
        if not stage1.empty:
            plt.plot(
                stage1["epoch"],
                stage1["acc"],
                marker="o",
                linestyle="-",
                color="tab:blue",
                label="Stage 1 (Representation Learning)",
            )

        # Plot Stage2 in orange, shifted to follow Stage1
        if not stage2.empty:
            offset = stage1["epoch"].max() if not stage1.empty else 0
            stage2_shifted = stage2.copy()
            stage2_shifted["epoch"] = stage2_shifted["epoch"] + offset

            plt.plot(
                stage2_shifted["epoch"],
                stage2_shifted["acc"],
                marker="o",
                linestyle="-",
                color="tab:orange",
                label="Stage 2 (Dynamic Adaptation)",
            )

            # Connect Stage1 last point to Stage2 first point with orange line
            if not stage1.empty:
                last_stage1_x = stage1["epoch"].max()
                last_stage1_y = stage1["acc"].iloc[-1]
                first_stage2_x = stage2_shifted["epoch"].iloc[0]
                first_stage2_y = stage2_shifted["acc"].iloc[0]
                plt.plot(
                    [last_stage1_x, first_stage2_x],
                    [last_stage1_y, first_stage2_y],
                    color="tab:orange",
                    linestyle="-",
                    linewidth=2,
                )

                # Add vertical dashed line at the transition
                plt.axvline(
                    x=last_stage1_x,
                    color="gray",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.8,
                )

        # Titles and labels
        model_display = model_name_map.get(model, model)
        scenario_display = scenario_name_map.get(scenario, scenario)
        plt.title(f"CIFAR-10 – {model_display} – {scenario_display}",
                  fontsize=11, fontweight="bold")

        plt.xlabel("Epoch (Stage 1 + Stage 2)")
        plt.ylabel("Accuracy")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=8)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{model}_{scenario}_acc_curves_best.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[plots] Saved enhanced accuracy curve for {model}, {scenario} → {out_path}")


# ---------------------------------------------------------
# 5. Best summary table and heatmap
# ---------------------------------------------------------

def save_best_summary_table(best_summary: pd.DataFrame,
                            out_path: str = "figures/best_summary.csv") -> None:
    """
    Save the best (model, scenario) summary table to CSV.
    """
    directory = os.path.dirname(out_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    best_summary.to_csv(out_path, index=False)
    print(f"[plots] Best summary table saved to: {out_path}")


def plot_metric_heatmap(best_summary: pd.DataFrame,
                        metric: str = "tail_macro_f1",
                        save_path: str = "figures/heatmap_tail_macro_f1.png"):
    """
    Plot a Tail Macro-F1 heatmap (higher values = darker blue).
    - Colormap: 'Blues' (light blue = low, dark blue = high)
    - White numeric labels centered in each cell.
    """
    _ensure_output_dir(save_path)

    df = best_summary.copy()
    df["model"] = df["model"].str.lower()
    model_order = ["resnet10", "resnet15", "resnet20"]
    scenario_order = ["head_tail", "long_tail"]

    pivot = df.pivot(index="model", columns="scenario", values=metric)
    pivot = pivot.reindex(index=model_order, columns=scenario_order)

    plt.figure(figsize=(6, 4))
    # Use normal 'Blues' so high = darker, low = lighter
    im = plt.imshow(pivot.values, aspect="auto", cmap="Blues", vmin=0.5, vmax=0.85)

    plt.xticks(range(len(scenario_order)), scenario_order)
    plt.yticks(range(len(model_order)), model_order)
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Tail Macro-F1")

    # Add white numbers in the center of each cell
    for i in range(len(model_order)):
        for j in range(len(scenario_order)):
            value = pivot.iloc[i, j]
            plt.text(j, i, f"{value:.3f}",
                     ha="center", va="center",
                     color="white", fontsize=10, fontweight="bold")

    plt.title("Tail Macro-F1 for each model and scenario (best epoch)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[plots] Tail Macro-F1 heatmap (dark=high) saved to: {save_path}")





# ---------------------------------------------------------
# 6. Class count table for long_tail and head_tail
# ---------------------------------------------------------

def compute_img_num_per_cls(
    imbalance_type: str,
    imb_factor: float = 0.01,
    num_classes: int = 10,
    img_max: int = 5000,
):
    """
    Compute per-class sample counts following the logic in data_loader.py:

        - long_tail:
            img_max * (imb_factor ** (cls_idx / (num_classes - 1)))

        - head_tail:
            first half classes  -> img_max
            second half classes -> geometric decay with the same formula
    """
    img_num_per_cls = []

    if imbalance_type == "balanced":
        img_num_per_cls = [img_max] * num_classes

    elif imbalance_type == "long_tail":
        for cls_idx in range(num_classes):
            num = img_max * (imb_factor ** (cls_idx / (num_classes - 1)))
            img_num_per_cls.append(int(num))

    elif imbalance_type == "head_tail":
        half = num_classes // 2
        for cls_idx in range(num_classes):
            if cls_idx < half:
                img_num_per_cls.append(img_max)
            else:
                tail_idx = cls_idx - half
                tail_classes = num_classes - half
                num = img_max * (imb_factor ** (tail_idx / max(tail_classes - 1, 1)))
                img_num_per_cls.append(int(num))
    else:
        raise ValueError(f"Unknown imbalance_type: {imbalance_type}")

    return img_num_per_cls


def make_class_count_table(
    imb_factor: float = 0.01,
    num_classes: int = 10,
    img_max: int = 5000,
) -> pd.DataFrame:
    """
    Build a table showing per-class retained data for:
        - long_tail
        - head_tail

    Returns:
        DataFrame with columns: ['class', 'long_tail', 'head_tail']
    """
    lt = compute_img_num_per_cls("long_tail", imb_factor, num_classes, img_max)
    ht = compute_img_num_per_cls("head_tail", imb_factor, num_classes, img_max)

    df = pd.DataFrame({
        "class": list(range(num_classes)),
        "long_tail": lt,
        "head_tail": ht,
    })
    return df


def save_class_count_table(
    out_path: str = "figures/class_counts_imb01.csv",
    imb_factor: float = 0.01,
    num_classes: int = 10,
    img_max: int = 5000,
):
    """
    Save the class count table to CSV.
    """
    directory = os.path.dirname(out_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    df = make_class_count_table(imb_factor, num_classes, img_max)
    df.to_csv(out_path, index=False)
    print(f"[plots] Class count table saved to: {out_path}")


# ---------------------------------------------------------
# 7. Main visualization pipeline
# ---------------------------------------------------------

def main_visualize(results_dir: str = ".", out_dir: str = "figures"):
    """
    Run a full visualization pipeline:

      1) Load all results_*.csv
      2) Summarize the best epoch per (model, scenario) by macro_f1
      3) Plot grouped bar charts for:
            - Accuracy
            - Macro F1
            - Tail Macro F1
      4) Plot a line chart for Macro F1 vs model depth
      5) Save best summary table and best hyper-parameter table
      6) Plot accuracy curves for the best run of each (model, scenario)
      7) Save the class count table for long_tail and head_tail
      8) Plot a heatmap for model × scenario × metric
    """
    os.makedirs(out_dir, exist_ok=True)

    print(f"[plots] Loading results from: {results_dir}")
    all_results = load_all_results(results_dir=results_dir)

    best_summary = summarize_best_by_model_and_scenario(
        all_results, metric="macro_f1"
    )
    print("[plots] Best summary table (macro_f1 criterion):")
    print(best_summary)

    # Grouped bar charts (SimpleCNN is filtered inside plotting functions)
    plot_grouped_bar(
        best_summary,
        metric="acc",
        save_path=os.path.join(out_dir, "bar_models_vs_scenario_acc.png"),
    )
    plot_grouped_bar(
        best_summary,
        metric="macro_f1",
        save_path=os.path.join(out_dir, "bar_models_vs_scenario_macro_f1.png"),
    )
    plot_grouped_bar(
        best_summary,
        metric="tail_macro_f1",
        save_path=os.path.join(out_dir, "bar_models_vs_scenario_tail_macro_f1.png"),
    )

    # Depth trend line chart
    plot_depth_trend(
        best_summary,
        metric="macro_f1",
        save_path=os.path.join(out_dir, "line_depth_trend_macro_f1.png"),
    )

    # Save tables
    save_best_summary_table(
        best_summary,
        out_path=os.path.join(out_dir, "best_summary.csv"),
    )

    save_best_hparam_table(
        best_summary,
        out_path=os.path.join(out_dir, "best_hparams.csv"),
    )

    # Heatmap (SimpleCNN filtered inside)
    plot_metric_heatmap(
        best_summary,
        metric="macro_f1",
        save_path=os.path.join(out_dir, "heatmap_macro_f1.png"),
    )

    # Accuracy curves for best runs (SimpleCNN skipped inside)
    plot_best_accuracy_curves_for_all(
        best_summary,
        results_dir=results_dir,
        out_dir=os.path.join(out_dir, "curves_best"),
    )

    # Class count table
    save_class_count_table(
        out_path=os.path.join(out_dir, "class_counts_imb01.csv"),
        imb_factor=0.01,
        num_classes=10,
        img_max=5000,
    )

    print(f"[plots] All visualizations and tables saved under: {out_dir}")


if __name__ == "__main__":
    # Run full pipeline on current directory
    main_visualize(results_dir=".", out_dir="figures")
