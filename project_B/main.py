import torch
from data_loader import get_dataloaders
from model import SimpleCNN
from losses import compute_class_weights_from_counts
from train_eval import train_one_epoch, evaluate
import multiprocessing


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ------- 1. Load data -------
    train_loader, test_loader, img_num_per_cls = get_dataloaders(
        data_root="./data",
        batch_size=128,
        # num_workers is not passed; default is 0
        imbalance_type="long_tail",   # options: "head_tail" / "balanced"
        imb_factor=0.01,
    )

    print("Class counts:", img_num_per_cls)

    # ------- 2. Build model -------
    model = SimpleCNN(num_classes=10).to(device)

    # ------- 3. Build optimizer -------
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
    )

    # ------- 4. Compute class weights (optional) -------
    class_weights = compute_class_weights_from_counts(img_num_per_cls).to(device)

    # ------- 5. Train -------
    num_epochs = 3

    for epoch in range(1, num_epochs + 1):
        train_loss, stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            num_classes=10,
            loss_type="ce",           # or "focal"
            class_weights=class_weights,
        )

        acc, macro_f1, tail_macro_f1 = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            num_classes=10,
        )

        print(
            f"[Epoch {epoch}] "
            f"loss={train_loss:.4f} "
            f"acc={acc:.4f} "
            f"macro_f1={macro_f1:.4f} "
            f"tail_macro_f1={tail_macro_f1:.4f}"
        )


if __name__ == "__main__":
    multiprocessing.freeze_support()   # Safe multiprocessing startup for Windows
    main()
