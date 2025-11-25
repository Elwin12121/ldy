import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score

from losses import FocalLoss


# ---------------------------------------------------------
# 1. Train for one epoch
# ---------------------------------------------------------
def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    num_classes: int = 10,
    loss_type: str = "ce",           # "ce" or "focal"
    class_weights: torch.Tensor = None,
    focal_gamma: float = 2.0,
):
    """
    Train the model for a single epoch.

    Args:
        model        : neural network.
        loader       : training DataLoader.
        optimizer    : optimizer (e.g., SGD, Adam).
        device       : "cuda" or "cpu".
        num_classes  : number of classes.
        loss_type    : "ce" (cross entropy) or "focal".
        class_weights: optional class-wise weights (1D tensor).
                       - for "ce"   -> passed to CrossEntropyLoss
                       - for "focal"-> passed as alpha to FocalLoss
        focal_gamma  : focusing parameter for focal loss.

    Returns:
        avg_loss     : average training loss over all samples.
        stats        : dict with
                       - "class_correct": np.ndarray of shape (num_classes,)
                       - "class_total"  : np.ndarray of shape (num_classes,)
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    # per-class statistics for dynamic weighting
    class_correct = np.zeros(num_classes, dtype=np.int64)
    class_total = np.zeros(num_classes, dtype=np.int64)

    # build criterion
    if loss_type == "focal":
        # focal loss with optional alpha as class weights
        if class_weights is not None:
            alpha = class_weights.to(device)
        else:
            alpha = None
        criterion = FocalLoss(gamma=focal_gamma, alpha=alpha, reduction="mean")
    else:
        # standard cross-entropy, possibly with class weights
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # update per-class statistics
        preds = logits.argmax(dim=1)
        correct = preds.eq(labels)

        for c in range(num_classes):
            mask = (labels == c)
            class_total[c] += mask.sum().item()
            class_correct[c] += (correct & mask).sum().item()

    avg_loss = total_loss / max(total_samples, 1)

    stats = {
        "class_correct": class_correct,
        "class_total": class_total,
    }

    return avg_loss, stats


# ---------------------------------------------------------
# 2. Evaluate on validation / test set
# ---------------------------------------------------------
@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    num_classes: int = 10,
):
    """
    Evaluate the model on a given DataLoader.

    Returns:
        acc           : overall accuracy
        macro_f1      : macro-averaged F1 over all classes
        tail_macro_f1 : macro F1 over tail classes
                        (here we define tail classes as the last half)
    """
    model.eval()

    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        preds = logits.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # overall accuracy
    acc = (all_preds == all_labels).mean()

    # per-class F1 scores
    f1_per_class = f1_score(
        all_labels,
        all_preds,
        average=None,
        labels=list(range(num_classes)),
    )
    macro_f1 = f1_per_class.mean()

    # define tail classes as the last half classes
    tail_classes = list(range(num_classes // 2, num_classes))
    tail_macro_f1 = f1_per_class[tail_classes].mean()

    return acc, macro_f1, tail_macro_f1
