# baseline_experiments.py
#
# Run multiple baselines on CIFAR-10 long-tail:
#   - Models: SimpleCNN, ResNet10, ResNet50
#   - Losses: CE, Class-Weighted CE, Focal Loss
#
# For each (model, loss) combination, we train for a few epochs and
# report: loss, accuracy, macro F1, tail macro F1.


import torch

from data_loader import get_dataloaders
from model import get_model
from losses import compute_class_weights_from_counts
from train_eval import train_one_epoch, evaluate


def run_single_baseline(
    model_name: str,
    method_name: str,
    loss_type: str,          # "ce" or "focal"
    use_class_weights: bool, # True / False
    num_epochs: int = 5,
):
    """
    Train a single (model, loss) configuration and print metrics.

    model_name    : "simplecnn" / "resnet10" / "resnet50"
    method_name   : e.g. "CE", "ClassWeighted CE", "Focal"
    loss_type     : "ce" or "focal" (passed to train_one_epoch)
    use_class_weights : whether to use static class weights
    num_epochs    : training epochs
    """
    print("\n==================================================")
    print(f"Model: {model_name} | Method: {method_name}")
    print("==================================================")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ----- Load long-tail CIFAR-10 -----
    train_loader, test_loader, img_num_per_cls = get_dataloaders(
        data_root="./data",
        batch_size=128,
        imbalance_type="long_tail",
        imb_factor=0.01,
    )
    print("Class counts:", img_num_per_cls)

    # ----- Build model -----
    model = get_model(model_name, num_classes=10, pretrained_resnet=False).to(device)

    # ----- Optimizer -----
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
    )

    # ----- Class weights (optional) -----
    class_weights = None
    if use_class_weights:
        class_weights = compute_class_weights_from_counts(img_num_per_cls).to(device)
        print("Using static class weights.")

    # ----- Training loop -----
    for epoch in range(1, num_epochs + 1):
        train_loss, stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            num_classes=10,
            loss_type=loss_type,        # "ce" or "focal"
            class_weights=class_weights,
        )

        acc, macro_f1, tail_macro_f1 = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            num_classes=10,
        )

        print(
            f"[{method_name}][{model_name}][Epoch {epoch}] "
            f"loss={train_loss:.4f} acc={acc:.4f} "
            f"macro_f1={macro_f1:.4f} tail_macro_f1={tail_macro_f1:.4f}"
        )


def run_all_baselines():
    """
    Run baselines for:
      - Models: simplecnn, resnet10, resnet50
      - Methods: CE, ClassWeighted CE, Focal
    """
    models = ["simplecnn", "resnet10", "resnet50"]

    # Define baseline methods
    methods = [
        {
            "method_name": "CE",
            "loss_type": "ce",
            "use_class_weights": False,
        },
        {
            "method_name": "ClassWeighted CE",
            "loss_type": "ce",
            "use_class_weights": True,
        },
        {
            "method_name": "Focal Loss",
            "loss_type": "focal",
            "use_class_weights": False,
        },
    ]

    for model_name in models:
        for cfg in methods:
            run_single_baseline(
                model_name=model_name,
                method_name=cfg["method_name"],
                loss_type=cfg["loss_type"],
                use_class_weights=cfg["use_class_weights"],
                num_epochs=5,   # you can adjust epochs here
            )


if __name__ == "__main__":
    run_all_baselines()
