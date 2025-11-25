import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision import datasets, transforms

# ---------- Load CIFAR-10 ----------
transform = transforms.ToTensor()
dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
class_names = dataset.classes  # ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# ---------- Select one image per class ----------
sample_images = {}
for idx, (img, label) in enumerate(dataset):
    if label not in sample_images:
        sample_images[label] = img
    if len(sample_images) == 10:
        break

# ---------- Display on a 2x5 grid ----------
plt.figure(figsize=(10, 5))
for i, label in enumerate(sorted(sample_images.keys())):
    plt.subplot(2, 5, i + 1)
    plt.imshow(np.transpose(sample_images[label].numpy(), (1, 2, 0)))
    plt.title(class_names[label], fontsize=10)
    plt.axis("off")
plt.tight_layout()
plt.savefig("cifar10_samples_2x5.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------- Construct imbalanced class distributions ----------
# Head–Tail scenario: 4 head classes (5000), 3 medium classes (400), 3 tail classes (10)
head_classes = [0, 1, 9, 8]
mid_classes = [3, 5, 7]
tail_classes = [2, 4, 6]
headtail_counts = [5000 if c in head_classes else 400 if c in mid_classes else 10 for c in range(10)]

# Long-tail scenario: exponential decay with r=0.5
base = 5000
r = 0.5
longtail_counts = [int(round(base * (r ** i))) for i in range(10)]

# ---------- Plot bar charts ----------
def plot_distribution(counts, title, save_path):
    plt.figure(figsize=(8, 4))
    plt.bar(class_names, counts, color="#4C72B0")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of Samples")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    for i, v in enumerate(counts):
        plt.text(i, v + max(counts)*0.02, str(v), ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

plot_distribution(headtail_counts, "Head–Tail Imbalance Scenario", "headtail_distribution.png")
plot_distribution(longtail_counts, "Long-Tail Imbalance Scenario (r=0.5)", "longtail_distribution.png")
