import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ---------- 1. Define image augmentations ----------
def build_transforms():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])

    return transform_train, transform_test


# ---------- 2. Build imbalanced indices ----------
def make_imbalanced_indices(targets, imbalance_type, imb_factor, num_classes):
    """
    Construct training indices based on the specified imbalance type.

    Parameters:
        targets        : list of original training labels
        imbalance_type : "balanced" / "long_tail" / "head_tail"
        imb_factor     : imbalance ratio (minority / majority), e.g. 0.01
        num_classes    : number of classes (10 for CIFAR-10)
    """
    targets = np.array(targets)
    img_max = len(targets) // num_classes  # Max samples per class (~5000)

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
                img_num_per_cls.append(img_max)  # Head classes: keep full samples
            else:
                tail_idx = cls_idx - half
                tail_classes = num_classes - half
                num = img_max * (imb_factor ** (tail_idx / max(tail_classes - 1, 1)))
                img_num_per_cls.append(int(num))
    else:
        raise ValueError(f"Unknown imbalance_type: {imbalance_type}")

    # Sample indices based on per-class desired counts
    new_indices = []
    for cls_idx, num in enumerate(img_num_per_cls):
        cls_indices = np.where(targets == cls_idx)[0]
        np.random.shuffle(cls_indices)
        selected = cls_indices[:num]
        new_indices.extend(selected.tolist())

    np.random.shuffle(new_indices)
    return new_indices, img_num_per_cls


# ---------- 3. Main function: return DataLoader ----------
def get_dataloaders(
    data_root="./data",
    batch_size=128,
    num_workers=0,                 # Recommended 0 for Windows
    imbalance_type="long_tail",
    imb_factor=0.01,
):
    num_classes = 10
    transform_train, transform_test = build_transforms()

    # Load the original full training dataset
    full_train = datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform_train,
    )

    # Keep the test set balanced
    test_set = datasets.CIFAR10(
        root=data_root,
        train=False,
        download=True,
        transform=transform_test,
    )

    # Build imbalanced training indices
    train_indices, img_num_per_cls = make_imbalanced_indices(
        targets=full_train.targets,
        imbalance_type=imbalance_type,
        imb_factor=imb_factor,
        num_classes=num_classes,
    )

    # Use subset as imbalanced training set
    train_set = Subset(full_train, train_indices)

    # Training DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    # Testing DataLoader
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, test_loader, img_num_per_cls
