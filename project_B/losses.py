import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------
# 1. Static class weights from class counts
# ---------------------------------------------------------
def compute_class_weights_from_counts(img_num_per_cls):
    """
    Compute inverse-frequency class weights from class sample counts.

    Args:
        img_num_per_cls: list or 1D array with number of samples per class.

    Returns:
        weights: torch.FloatTensor of shape (num_classes,).
    """
    counts = np.array(img_num_per_cls, dtype=np.float32)  # [C]

    # Avoid division by zero
    inv = 1.0 / (counts + 1e-6)      # Lower frequency -> higher weight
    inv = inv / inv.mean()           # Normalize to mean ~1 to prevent extreme imbalance

    weights = torch.tensor(inv, dtype=torch.float32)
    return weights


# ---------------------------------------------------------
# 2. Focal Loss
# ---------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Args:
        gamma: focusing parameter (default 2.0).
        alpha: None, float, or 1D tensor (per-class weights).
        reduction: "mean", "sum", or "none".

    Usage:
        criterion = FocalLoss(gamma=2.0, alpha=class_weights_tensor)
        loss = criterion(logits, targets)
    """

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes) raw outputs from the network.
            targets: (B,) ground-truth class indices.

        Returns:
            scalar loss (if reduction is "mean" or "sum")
            or per-sample loss (if reduction is "none").
        """
        # standard cross-entropy (per-sample)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # pt = probability of the true class
        pt = torch.exp(-ce_loss)

        # focal term (1 - pt)^gamma
        focal_term = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            # support per-class alpha
            if isinstance(self.alpha, (list, tuple, torch.Tensor)):
                alpha_t = torch.as_tensor(self.alpha, dtype=torch.float32, device=logits.device)
                alpha_t = alpha_t[targets]   # pick weight for each target class
            else:
                # scalar alpha
                alpha_t = torch.tensor(self.alpha, dtype=torch.float32, device=logits.device)

            loss = alpha_t * focal_term * ce_loss
        else:
            loss = focal_term * ce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:   # "none"
            return loss


# ---------------------------------------------------------
# 3. ICTTA-like dynamic class weights based on performance
# ---------------------------------------------------------
def compute_dynamic_weights_from_performance(
    class_correct,
    class_total,
    num_classes: int,
    min_weight: float = 1.0,
    max_weight: float = 5.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Dynamically update class weights based on each class's performance
    (correct / total) in the previous training round.
    Idea: worse performance (lower accuracy) -> higher weight,
    better performance -> lower weight.

    Inputs can be list / numpy.ndarray / torch.Tensor.
    Internally converted to torch.Tensor.
    """
    # Convert to tensor
    class_correct = torch.as_tensor(class_correct, dtype=torch.float32)
    class_total   = torch.as_tensor(class_total,   dtype=torch.float32)

    # Avoid division by zero
    class_total = torch.clamp(class_total, min=1.0)

    # Per-class accuracy
    acc = class_correct / (class_total + eps)   # range 0~1

    # Inverse proportional weighting: lower accuracy -> higher 1/acc
    inv = 1.0 / (acc + eps)

    # Normalize to [0, 1]
    inv_norm = inv / (inv.max() + eps)

    # Map to [min_weight, max_weight]
    weights = min_weight + (max_weight - min_weight) * inv_norm

    return weights
