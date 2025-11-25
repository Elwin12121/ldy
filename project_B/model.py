import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock, ResNet as TVResNet

# ============================================================
# 1. SimpleCNN baseline model
# ============================================================

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward_features(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x)
        return x


# ============================================================
# 2. ResNet-10 for CIFAR
# ============================================================

class ResNet10ForCIFAR(nn.Module):
    """
    ResNet10 backbone + classifier for CIFAR-10.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        base = TVResNet(block=BasicBlock, layers=[1, 1, 1, 1])
        self.feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward_features(self, x):
        x = self.backbone(x)
        return x

    def forward(self, x):
        feats = self.forward_features(x)
        logits = self.classifier(feats)
        return logits


# ============================================================
# 2.5 ResNet-15 for CIFAR
# ============================================================

class ResNet15ForCIFAR(nn.Module):
    """
    ResNet-15 style backbone + classifier for CIFAR-10.

    Notes:
      - Uses BasicBlock, layers = [1, 1, 2, 2]
      - Depth lies between ResNet-10 (1,1,1,1) and ResNet-20 (2,2,2,2)
      - Interface is consistent with ResNet10ForCIFAR / ResNet20ForCIFAR:
          - self.backbone
          - self.classifier
          - forward_features(x)
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Backbone: implemented using torchvision's ResNet
        base = TVResNet(block=BasicBlock, layers=[1, 1, 2, 2])

        # Final feature dimension (usually 512)
        self.feat_dim = base.fc.in_features

        # Remove original fc, keep only the feature extractor
        base.fc = nn.Identity()

        self.backbone = base
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward_features(self, x):
        """
        Output a feature vector of shape [B, feat_dim], used in two-stage Stage2.
        """
        x = self.backbone(x)
        return x

    def forward(self, x):
        feats = self.forward_features(x)
        logits = self.classifier(feats)
        return logits


# ============================================================
# 3. ResNet-20 for CIFAR
# ============================================================

class ResNet20ForCIFAR(nn.Module):
    """
    ResNet-20 backbone + classifier for CIFAR-10.
    layers = [2, 2, 2, 2]
    """
    def __init__(self, num_classes=10):
        super().__init__()
        base = TVResNet(block=BasicBlock, layers=[2, 2, 2, 2])
        self.feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.classifier = nn.Linear(self.feat_dim, num_classes)

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        feats = self.forward_features(x)
        logits = self.classifier(feats)
        return logits


# ============================================================
# 4. Utility functions
# ============================================================

def freeze_backbone(model):
    """
    Freeze the backbone parameters (used for Stage2).
    """
    if hasattr(model, "backbone"):
        for param in model.backbone.parameters():
            param.requires_grad = False


def get_classifier_params(model):
    """
    Return only the parameters of the classifier layer.
    """
    if hasattr(model, "classifier"):
        return model.classifier.parameters()
    else:
        return model.parameters()


def get_model(name: str, num_classes: int = 10, pretrained_resnet: bool = False):
    """
    Factory function to create models by name.
    Supported:
      - simplecnn
      - resnet10
      - resnet15
      - resnet20
    """
    name = name.lower()
    if name == "simplecnn":
        return SimpleCNN(num_classes=num_classes)
    elif name == "resnet10":
        return ResNet10ForCIFAR(num_classes=num_classes)
    elif name == "resnet15":
        return ResNet15ForCIFAR(num_classes=num_classes)
    elif name == "resnet20":
        return ResNet20ForCIFAR(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {name}")
