import torch
from torch import nn
from torch.nn import functional as F

from bayesian_layers import BayesianLinear


class StandardMLP(nn.Module):
    def __init__(self, hidden_dim: int = 400, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DropoutMLP(nn.Module):
    def __init__(self, hidden_dim: int = 400, dropout: float = 0.5, num_classes: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BayesianMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 400,
        num_classes: int = 10,
        dropout: float = 0.0,
        **layer_kwargs,
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = BayesianLinear(28 * 28, hidden_dim, **layer_kwargs)
        self.fc2 = BayesianLinear(hidden_dim, hidden_dim, **layer_kwargs)
        self.fc3 = BayesianLinear(hidden_dim, num_classes, **layer_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def log_prior(self) -> torch.Tensor:
        return self.fc1.log_p + self.fc2.log_p + self.fc3.log_p

    def log_variational_posterior(self) -> torch.Tensor:
        return self.fc1.log_q + self.fc2.log_q + self.fc3.log_q


class RegressionMLP(nn.Module):
    def __init__(self, hidden_dim: int = 100) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BayesianRegressionMLP(nn.Module):
    def __init__(self, hidden_dim: int = 100, **layer_kwargs) -> None:
        super().__init__()
        self.fc1 = BayesianLinear(1, hidden_dim, **layer_kwargs)
        self.fc2 = BayesianLinear(hidden_dim, hidden_dim, **layer_kwargs)
        self.fc3 = BayesianLinear(hidden_dim, 1, **layer_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def log_prior(self) -> torch.Tensor:
        return self.fc1.log_p + self.fc2.log_p + self.fc3.log_p

    def log_variational_posterior(self) -> torch.Tensor:
        return self.fc1.log_q + self.fc2.log_q + self.fc3.log_q
