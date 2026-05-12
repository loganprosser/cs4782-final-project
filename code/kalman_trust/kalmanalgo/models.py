from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, hidden_size: int = 400, dropout: bool = False, dropout_p: float = 0.5) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_size),
            nn.ReLU(),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout_p))
        layers.extend(
            [
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            ]
        )
        if dropout:
            layers.append(nn.Dropout(dropout_p))
        layers.append(nn.Linear(hidden_size, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleCNN(nn.Module):
    def __init__(self, hidden_size: int = 128, dropout: bool = False, dropout_p: float = 0.25) -> None:
        super().__init__()
        conv_layers: list[nn.Module] = [
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ]
        if dropout:
            conv_layers.append(nn.Dropout2d(dropout_p))
        conv_layers.extend(
            [
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
        )
        if dropout:
            conv_layers.append(nn.Dropout2d(dropout_p))
        self.features = nn.Sequential(*conv_layers)

        classifier: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, hidden_size),
            nn.ReLU(),
        ]
        if dropout:
            classifier.append(nn.Dropout(dropout_p))
        classifier.append(nn.Linear(hidden_size, 10))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(model: str, method: str, hidden_size: int = 400, dropout_p: float | None = None) -> nn.Module:
    use_dropout = method in {"dropout", "dropout_kalman"}
    if model == "mlp":
        return MLP(hidden_size=hidden_size, dropout=use_dropout, dropout_p=0.5 if dropout_p is None else dropout_p)
    if model == "cnn":
        cnn_hidden = hidden_size if hidden_size > 0 else 128
        return SimpleCNN(hidden_size=cnn_hidden, dropout=use_dropout, dropout_p=0.25 if dropout_p is None else dropout_p)
    raise ValueError(f"Unsupported model: {model}")


def uses_kalman(method: str) -> bool:
    return method in {"kalman", "dropout_kalman"}
