"""
two_stage_ictta_v3.py
------------------------------------------------------------
Final two-stage training framework for CIFAR-10 imbalance.

Stage 1 (Representation Learning)
    • Train backbone + classifier on a static imbalanced dataset
      ("long_tail" or "head_tail")
    • Standard Cross-Entropy, no class re-weighting
    • Goal: learn strong feature representations

Stage 2 (Dynamic Adaptation, ICTTA-style)
    • Freeze backbone (ResNet-style models) or train all params (SimpleCNN)
    • Re-train on the SAME scenario as Stage 1
    • Update per-class weights each epoch based on
        – accuracy  (dynamic_mode="acc"), or
        – mean loss (dynamic_mode="loss")
    • Optionally use Focal Loss to emphasize hard / tail samples
------------------------------------------------------------
"""

import os
from typing import Optional
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_loader import get_dataloaders
from model import get_model, freeze_backbone, get_classifier_params
from losses import compute_dynamic_weights_from_performance, FocalLoss
from train_eval import train_one_epoch, evaluate
from results_logger import log_epoch_result


# ============================================================
# Stage 1 – Representation Learning
# ============================================================
def stage1_train(
    model: nn.Module,
    train_loader,
    test_loader,
    device: str,
    num_epochs: int,
    log_file: str,
    model_name_for_log: str,
    scenario_name_for_log: str,
    lr: float = 0.1,
) -> nn.Module:
    """Train full network with plain Cross-Entropy."""
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

    print("\n===== Stage 1: Representation Learning =====")
    for epoch in range(1, num_epochs + 1):
        train_loss, stats = train_one_epoch(
            model, train_loader, optimizer, device,
            num_classes=10, loss_type="ce", class_weights=None,
        )

        acc, macro_f1, tail_macro_f1 = evaluate(
            model, test_loader, device, num_classes=10
        )

        print(f"[Stage1][{scenario_name_for_log}][{model_name_for_log}] "
              f"Epoch {epoch:02d} | loss={train_loss:.4f} "
              f"acc={acc:.4f} macro_f1={macro_f1:.4f} "
              f"tail_macro_f1={tail_macro_f1:.4f}")

        log_epoch_result(
            filename=log_file,
            stage=f"stage1_{scenario_name_for_log}",
            epoch=epoch,
            model_name=model_name_for_log,
            method="two_stage_ictta_v3",
            metrics=dict(loss=train_loss, acc=acc,
                         macro_f1=macro_f1, tail_macro_f1=tail_macro_f1),
        )
        scheduler.step()
    return model


# ============================================================
# Helper – Compute loss-based dynamic weights
# ============================================================
@torch.no_grad()
def compute_loss_based_weights(
    model: nn.Module,
    train_loader,
    device: str,
    num_classes: int = 10,
    min_weight: float = 1.0,
    max_weight: float = 8.0,
    loss_type: str = "focal",
    focal_gamma: float = 2.0,
) -> torch.Tensor:
    """Compute per-class mean loss and map to [min,max] → weights."""
    model.eval()
    loss_sum = torch.zeros(num_classes, device=device)
    count_sum = torch.zeros(num_classes, device=device)

    criterion = (FocalLoss(gamma=focal_gamma, alpha=None, reduction="none")
                 if loss_type == "focal"
                 else nn.CrossEntropyLoss(reduction="none"))

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        losses = criterion(model(images), labels)
        for c in range(num_classes):
            mask = (labels == c)
            if mask.any():
                loss_sum[c] += losses[mask].sum()
                count_sum[c] += mask.sum()

    mean_loss = loss_sum / (count_sum + 1e-6)
    norm = mean_loss / (mean_loss.max() + 1e-6)
    return min_weight + (max_weight - min_weight) * norm


# ============================================================
# Stage 2 – Dynamic Adaptation
# ============================================================
def stage2_train_dynamic(
    model: nn.Module,
    train_loader,
    test_loader,
    device: str,
    num_epochs: int,
    min_weight: float,
    max_weight: float,
    log_file: str,
    model_name_for_log: str,
    scenario_name_for_log: str,
    lr: float = 0.01,
    ema_alpha: float = 0.8,
    loss_type: str = "focal",
    focal_gamma: float = 2.0,
    dynamic_mode: str = "loss",
) -> nn.Module:
    """Freeze backbone (if exists) and perform ICTTA-style re-training."""
    print("\n===== Stage 2: Dynamic Adaptation (ICTTA-style) =====")
    print(f"Scenario={scenario_name_for_log}, mode={dynamic_mode}, "
          f"loss={loss_type}, gamma={focal_gamma}, ema={ema_alpha}")

    if hasattr(model, "backbone"):
        print("Freezing backbone parameters.")
        freeze_backbone(model)
        params = get_classifier_params(model)
    else:
        print("No backbone – training all parameters.")
        params = model.parameters()

    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    dynamic_weights = None

    for epoch in range(1, num_epochs + 1):
        train_loss, stats = train_one_epoch(
            model, train_loader, optimizer, device,
            num_classes=10, loss_type=loss_type,
            class_weights=dynamic_weights, focal_gamma=focal_gamma,
        )

        # Update class weights
        if dynamic_mode == "acc":
            new_w = compute_dynamic_weights_from_performance(
                stats["class_correct"], stats["class_total"], 10,
                min_weight, max_weight).to(device)
        elif dynamic_mode == "loss":
            new_w = compute_loss_based_weights(
                model, train_loader, device, 10,
                min_weight, max_weight, loss_type, focal_gamma).to(device)
        else:
            raise ValueError(f"Unknown dynamic_mode {dynamic_mode}")

        dynamic_weights = new_w if dynamic_weights is None \
            else ema_alpha * dynamic_weights + (1 - ema_alpha) * new_w

        acc, macro_f1, tail_macro_f1 = evaluate(model, test_loader, device, 10)
        print(f"[Stage2][{scenario_name_for_log}][{model_name_for_log}] "
              f"Epoch {epoch:02d} | loss={train_loss:.4f} acc={acc:.4f} "
              f"macro_f1={macro_f1:.4f} tail_macro_f1={tail_macro_f1:.4f}")

        log_epoch_result(
            filename=log_file,
            stage=f"stage2_{scenario_name_for_log}",
            epoch=epoch,
            model_name=model_name_for_log,
            method="two_stage_ictta_v3",
            metrics=dict(loss=train_loss, acc=acc,
                         macro_f1=macro_f1, tail_macro_f1=tail_macro_f1),
        )
        scheduler.step()
    return model


# ============================================================
# Run one model × one scenario
# ============================================================
def run_two_stage_ictta_on_scenario(
    model_name: str,
    imbalance_type: str,
    stage1_epochs: int,
    stage2_epochs: int,
    imb_factor: float,
    batch_size: int,
    log_dir: str,
    stage1_lr: float,
    stage2_lr: float,
    min_weight: float,
    max_weight: float,
    ema_alpha: float,
    stage2_loss_type: str,
    stage2_focal_gamma: float,
    dynamic_mode: str,
    exp_name: Optional[str] = None,
) -> None:
    """Full two-stage pipeline for one (model, scenario) pair."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scenario = imbalance_type
    log_file = (os.path.join(log_dir, f"results_{exp_name}.csv")
                if exp_name else
                os.path.join(log_dir, f"results_{model_name}_{scenario}.csv"))

    print("\n============================================================")
    print(f"Run: {model_name} on {scenario}  →  log: {log_file}")
    print("============================================================")

    train_loader, test_loader, counts = get_dataloaders(
        data_root="./data", batch_size=batch_size,
        imbalance_type=imbalance_type, imb_factor=imb_factor)
    print(f"[{scenario}] class counts:", counts)

    model = get_model(model_name, 10, pretrained_resnet=False).to(device)

    model = stage1_train(
        model, train_loader, test_loader, device, stage1_epochs,
        log_file, model_name, scenario, stage1_lr)

    stage2_train_dynamic(
        model, train_loader, test_loader, device, stage2_epochs,
        min_weight, max_weight, log_file, model_name, scenario,
        stage2_lr, ema_alpha, stage2_loss_type,
        stage2_focal_gamma, dynamic_mode)

    print(f"\nFinished model={model_name}, scenario={scenario}")


# ============================================================
# Run all 4 models × 2 scenarios
# ============================================================
def run_all_models_two_scenarios() -> None:
    """Baseline grid: 4 models × 2 scenarios."""
    models = ["simplecnn", "resnet10", "resnet15", "resnet20"]
    scenarios = ["long_tail", "head_tail"]
    for m in models:
        for s in scenarios:
            run_two_stage_ictta_on_scenario(
                model_name=m, imbalance_type=s,
                stage1_epochs=15, stage2_epochs=20,
                imb_factor=0.01, batch_size=128, log_dir=".",
                stage1_lr=0.05 if m == "simplecnn" else 0.1,
                stage2_lr=0.01, min_weight=1.0, max_weight=8.0,
                ema_alpha=0.8, stage2_loss_type="focal",
                stage2_focal_gamma=1.5, dynamic_mode="loss",
                exp_name=f"baseline_{m}_{s}",
            )


if __name__ == "__main__":
    run_all_models_two_scenarios()
