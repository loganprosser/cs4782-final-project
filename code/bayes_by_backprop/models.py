import torch
from torch import nn
from torch.nn import functional as F

from .bayesian_layers import BayesianLinear


class StandardMLP(nn.Module):
    def __init__(self, hidden_dim: int = 400, num_classes: int = 10, hidden_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        in_features = 28 * 28
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(in_features, hidden_dim), nn.ReLU()])
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DropoutMLP(nn.Module):
    def __init__(self, hidden_dim: int = 400, dropout: float = 0.5, num_classes: int = 10, hidden_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        in_features = 28 * 28
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(in_features, hidden_dim), nn.ReLU(), nn.Dropout(dropout)])
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BayesianMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 400,
        num_classes: int = 10,
        dropout: float = 0.0,
        hidden_layers: int = 2,
        **layer_kwargs,
    ) -> None:
        super().__init__()
        self.hidden_layers = hidden_layers
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        if hidden_layers == 2:
            self.fc1 = BayesianLinear(28 * 28, hidden_dim, **layer_kwargs)
            self.fc2 = BayesianLinear(hidden_dim, hidden_dim, **layer_kwargs)
            self.fc3 = BayesianLinear(hidden_dim, num_classes, **layer_kwargs)
        else:
            layers = []
            in_features = 28 * 28
            for _ in range(hidden_layers):
                layers.append(BayesianLinear(in_features, hidden_dim, **layer_kwargs))
                in_features = hidden_dim
            self.hidden = nn.ModuleList(layers)
            self.out = BayesianLinear(in_features, num_classes, **layer_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        if self.hidden_layers == 2:
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            return self.fc3(x)
        for layer in self.hidden:
            x = F.relu(layer(x))
            x = self.dropout(x)
        return self.out(x)

    def log_prior(self) -> torch.Tensor:
        if self.hidden_layers == 2:
            return self.fc1.log_p + self.fc2.log_p + self.fc3.log_p
        return sum((layer.log_p for layer in self.hidden), self.out.log_p)

    def log_variational_posterior(self) -> torch.Tensor:
        if self.hidden_layers == 2:
            return self.fc1.log_q + self.fc2.log_q + self.fc3.log_q
        return sum((layer.log_q for layer in self.hidden), self.out.log_q)


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
