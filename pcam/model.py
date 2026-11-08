import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(pretrained: bool) -> nn.Module:
    """Standard torchvision ResNet-18 with a single-logit head (binary tumour vs. normal)."""
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    m.fc = nn.Linear(m.fc.in_features, 1)
    return m
