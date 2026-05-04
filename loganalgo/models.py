from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


MODEL_NAMES = [
    "mlp",
    "mlp_wide",
    "mlp_deep",
    "cnn",
    "cnn_deep",
    "cnn_bn",
    "resnet_tiny",
]


class StandardMLP(nn.Module):
    def __init__(self, hidden_dim: int = 400, num_classes: int = 10) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        features = F.relu(self.fc2(x))
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


class WideMLP(StandardMLP):
    def __init__(self, hidden_dim: int = 400, num_classes: int = 10) -> None:
        super().__init__(hidden_dim=hidden_dim * 2, num_classes=num_classes)


class DeepMLP(nn.Module):
    def __init__(self, hidden_dim: int = 400, depth: int = 4, num_classes: int = 10) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten(), nn.Linear(28 * 28, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


class SimpleCNN(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(64 * 7 * 7, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        features = F.relu(self.fc(x))
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


class DeepCNN(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(128 * 3 * 3, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        features = F.relu(self.fc(x))
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


class BatchNormCNN(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(64 * 7 * 7, hidden_dim)
        self.bn_fc = nn.BatchNorm1d(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        features = F.relu(self.bn_fc(self.fc(x)))
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class TinyResNet(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.layer1 = nn.Sequential(ResidualBlock(32, 32), ResidualBlock(32, 32))
        self.layer2 = nn.Sequential(ResidualBlock(32, 64, stride=2), ResidualBlock(64, 64))
        self.layer3 = nn.Sequential(ResidualBlock(64, 128, stride=2), ResidualBlock(128, 128))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = torch.flatten(self.pool(x), start_dim=1)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


def build_model(model_name: str, hidden_dim: int = 400) -> nn.Module:
    if model_name == "mlp":
        return StandardMLP(hidden_dim=hidden_dim)
    if model_name == "mlp_wide":
        return WideMLP(hidden_dim=hidden_dim)
    if model_name == "mlp_deep":
        return DeepMLP(hidden_dim=hidden_dim)
    if model_name == "cnn":
        return SimpleCNN(hidden_dim=hidden_dim)
    if model_name == "cnn_deep":
        return DeepCNN(hidden_dim=max(hidden_dim, 256))
    if model_name == "cnn_bn":
        return BatchNormCNN(hidden_dim=hidden_dim)
    if model_name == "resnet_tiny":
        return TinyResNet()
    raise ValueError(f"Unsupported model: {model_name}")
