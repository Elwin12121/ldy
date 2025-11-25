import os
import torch

from data_loader import get_dataloaders
from model import get_model
from losses import compute_class_weights_from_counts
from train_eval import train_one_epoch, evaluate
from results_logger import log_epoch_result
from plots import load_all_results, summarize_best_by_model_and_scenario


def run_single_stage_baseline(
    model_name: str,
    imbalance_type: str = "long_tail",    # "long_tail" or "head_tail"
    method: str = "ce",                   # "ce", "ce_class_weight", "focal"
    num_epochs: int = 50,
    lr_simplecnn: float = 0.05,
    lr_resnet: float = 0.1,
    focal_gamma: float = 1.5,
    log_dir: str = ".",
):
    """
    Run one single-stage baseline configuration.

    Args:
        model_name:      "simplecnn", "resnet10", "resnet15", or "resnet20".
        imbalance_type:  Imbalance scenario, "long_tail" or "head_tail".
        method:          Loss configuration: "ce", "ce_class_weight", "focal".
        num_epochs:      Number of training epochs.
        lr_simplecnn:    Learning rate for SimpleCNN.
        lr_resnet:       Learning rate for ResNet models.
        focal_gamma:     Gamma for focal loss (when method == "focal").
        log_dir:         Directory where result CSV files will be saved.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scenario = imbalance_type
    method = method.lower()

    print("\n============================================================")
    print(f"Single-Stage Baseline | model={model_name} | scenario={scenario} | method={method}")
    print("Device:", device)
    print("============================================================")

    # 1) Data loaders
    train_loader, test_loader, img_num_per_cls = get_dataloaders(
        data_root="./data",
        batch_size=128,
        imbalance_type=scenario,
        imb_factor=0.01,
    )
    print(f"[{scenario}] class counts:", img_num_per_cls)

    # 2) Model
    model = get_model(model_name, num_classes=10, pretrained_resnet=False).to(device)

    # 3) Optimizer
    if model_name.lower() == "simplecnn":
        lr = lr_simplecnn
    else:
        lr = lr_resnet

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    # 4) Loss configuration
    loss_type = "ce"
    class_weights = None

    if method == "ce":
        # Plain cross-entropy, no class weights
        loss_type = "ce"
        class_weights = None
        method_name_for_log = "single_ce"

    elif method == "ce_class_weight":
        # Cross-entropy with static inverse-frequency class weights
        loss_type = "ce"
        class_weights = compute_class_weights_from_counts(img_num_per_cls).to(device)
        method_name_for_log = "single_ce_class_weight"
        print("Using static class weights for CE.")

    elif method == "focal":
        # Focal loss without extra class weights
        loss_type = "focal"
        class_weights = None
        method_name_for_log = "single_focal"
        print(f"Using Focal Loss (gamma={focal_gamma}).")

    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Expected 'ce', 'ce_class_weight', or 'focal'."
        )

    # 5) Log file path
    exp_name = f"single_{model_name}_{scenario}_{method}"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"results_{exp_name}.csv")
    print("Logging to:", log_path)

    # 6) Training loop
    for epoch in range(1, num_epochs + 1):
        # Train for one epoch
        train_loss, stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            num_classes=10,
            loss_type=loss_type,
            class_weights=class_weights,
            focal_gamma=focal_gamma,
        )

        # Evaluate on the test set
        acc, macro_f1, tail_macro_f1 = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            num_classes=10,
        )

        print(
            f"[Baseline][{scenario}][{model_name}][{method_name_for_log}] "
            f"Epoch {epoch:02d} | "
            f"loss={train_loss:.4f} acc={acc:.4f} "
            f"macro_f1={macro_f1:.4f} tail_macro_f1={tail_macro_f1:.4f}"
        )

        # Log metrics to CSV
        # Note: stage = "baseline" to distinguish from two-stage runs
        log_epoch_result(
            filename=log_path,
            stage="baseline",
            epoch=epoch,
            model_name=model_name,
            method=method_name_for_log,
            metrics=dict(
                loss=train_loss,
                acc=acc,
                macro_f1=macro_f1,
                tail_macro_f1=tail_macro_f1,
            ),
        )


def summarize_baseline_best(
    results_dir: str = ".",
    out_csv_all: str = "baseline_best_single_stage.csv",
    out_csv_per_model: str = "baseline_best_per_model.csv",
):
    """
    Summarize the best single-stage baselines.

    Steps:
        1) Load all results_single_*.csv files using load_all_results().
        2) Use summarize_best_by_model_and_scenario() to pick the best
           epoch per (model, scenario) (at most 8 rows).
        3) From these 8 rows, select one best row per model w.r.t macro_f1
           (4 rows, one per model).

    Args:
        results_dir:       Directory containing results_single_*.csv.
        out_csv_all:       Path to save best rows per (model, scenario).
        out_csv_per_model: Path to save one best row per model.

    Returns:
        best_by_ms:   DataFrame, best row per (model, scenario).
        best_overall: DataFrame, one best row per model.
    """
    # Load only single-stage baseline results
    all_results = load_all_results(
        results_dir=results_dir,
        pattern="results_single_*.csv",
    )

    # Step 1: best per (model, scenario)
    best_by_ms = summarize_best_by_model_and_scenario(
        all_results,
        metric="macro_f1",
    )

    # Save the 8-row table (per model & scenario)
    if out_csv_all:
        os.makedirs(os.path.dirname(out_csv_all), exist_ok=True) if os.path.dirname(out_csv_all) else None
        best_by_ms.to_csv(out_csv_all, index=False)
        print(f"[baseline] Saved best per (model, scenario) → {out_csv_all}")
        print(best_by_ms)

    # Step 2: for each model, pick the overall best row across scenarios
    best_overall = (
        best_by_ms.sort_values("macro_f1", ascending=False)
        .groupby("model", as_index=False)
        .head(1)
        .sort_values("model")
        .reset_index(drop=True)
    )

    if out_csv_per_model:
        os.makedirs(os.path.dirname(out_csv_per_model), exist_ok=True) if os.path.dirname(out_csv_per_model) else None
        best_overall.to_csv(out_csv_per_model, index=False)
        print(f"[baseline] Saved best per model (4 rows) → {out_csv_per_model}")
        print(best_overall)

    return best_by_ms, best_overall


def plot_best_baseline_accuracy_curves(
    best_overall,
    results_dir: str = ".",
    out_dir: str = "figures/baseline_curves",
):
    """
    Plot accuracy-vs-epoch curves for the best single-stage baseline
    configuration of each model (4 curves in total).

    Args:
        best_overall: DataFrame with one row per model, as returned
                      by summarize_baseline_best().
        results_dir:  Directory where the original CSV logs are stored.
        out_dir:      Directory to save the curve figures.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)

    model_name_map = {
        "simplecnn": "SimpleCNN",
        "resnet10": "ResNet-10",
        "resnet15": "ResNet-15",
        "resnet20": "ResNet-20",
    }

    scenario_name_map = {
        "head_tail": "Head-Tail Imbalance",
        "long_tail": "Long-Tail Imbalance",
        "unknown": "Unknown Scenario",
    }

    for _, row in best_overall.iterrows():
        model = str(row["model"])
        scenario = str(row.get("scenario", "unknown"))
        source_file = str(row.get("source_file", ""))  # e.g. results_single_resnet10_long_tail_ce
        exp = str(row.get("exp", ""))

        csv_path = os.path.join(results_dir, f"{source_file}.csv")
        if not os.path.isfile(csv_path):
            print(f"[baseline] Warning: log file not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)

        # Filter rows for the correct model (usually all rows in this file)
        sub = df[df["model"].str.lower() == model.lower()].copy()
        if sub.empty:
            print(f"[baseline] No rows for model={model} in {csv_path}")
            continue

        sub = sub.sort_values("epoch")

        plt.figure(figsize=(6, 4))
        plt.plot(
            sub["epoch"],
            sub["acc"],
            marker="o",
            linestyle="-",
        )

        model_display = model_name_map.get(model, model)
        scenario_display = scenario_name_map.get(scenario, scenario)

        plt.title(
            f"CIFAR-10 – {model_display} – {scenario_display}\n"
            f"Best Single-Stage Baseline (exp: {exp})",
            fontsize=11,
        )
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"{model}_best_single_stage_acc_curve.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[baseline] Saved accuracy curve for best {model} → {out_path}")


def run_all_single_stage_baselines():
    """
    Run all single-stage baselines and then summarize best results.

    Workflow:
        1) Train single-stage baselines for:
               models   : simplecnn, resnet10, resnet15, resnet20
               scenarios: long_tail, head_tail
               methods  : CE, Class-Weighted CE, Focal Loss
           → 4 × 2 × 3 = 24 runs total, each producing a results_single_*.csv file.
        2) Summarize:
               - best per (model, scenario)       → baseline_best_single_stage.csv
               - best overall per model (4 rows)  → baseline_best_per_model.csv
        3) Plot accuracy curves only for these 4 best configurations.
    """
    models = ["simplecnn", "resnet10", "resnet15", "resnet20"]
    scenarios = ["long_tail", "head_tail"]
    methods = ["ce", "ce_class_weight", "focal"]

    for scenario in scenarios:
        for model_name in models:
            for method in methods:
                run_single_stage_baseline(
                    model_name=model_name,
                    imbalance_type=scenario,
                    method=method,
                    num_epochs=50,
                    lr_simplecnn=0.05,
                    lr_resnet=0.1,
                    focal_gamma=1.5,
                    log_dir=".",
                )

    # After all 24 runs are finished, summarize and plot
    best_by_ms, best_overall = summarize_baseline_best(
        results_dir=".",
        out_csv_all="baseline_best_single_stage.csv",
        out_csv_per_model="baseline_best_per_model.csv",
    )

    plot_best_baseline_accuracy_curves(
        best_overall=best_overall,
        results_dir=".",
        out_dir="figures/baseline_curves",
    )


if __name__ == "__main__":
    run_all_single_stage_baselines()
