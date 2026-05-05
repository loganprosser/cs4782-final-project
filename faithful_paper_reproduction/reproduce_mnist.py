from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import math
import multiprocessing as mp
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


HIDDEN_UNITS = (400, 800, 1200)
LEARNING_RATES = (1e-3, 1e-4, 1e-5)
TEST_MC_SAMPLES = (1, 2, 5, 10)
SCALE_MIXTURE_PI = (0.25, 0.5, 0.75)
SCALE_MIXTURE_NEG_LOG_SIGMA1 = (0, 1, 2)
SCALE_MIXTURE_NEG_LOG_SIGMA2 = (6, 7, 8)
PRUNING_PERCENTAGES = (0, 50, 75, 95, 98)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def config_seed(base_seed: int, config: "RunConfig") -> int:
    payload = repr(config).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return (base_seed + int(digest[:8], 16)) % (2**31)


def get_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def divide_by_126(x: torch.Tensor) -> torch.Tensor:
    return x.float().div(126.0)


def mnist_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.PILToTensor(),
            transforms.Lambda(divide_by_126),
        ]
    )


def build_data_loaders(
    data_dir: Path,
    batch_size: int,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    full_train = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=mnist_transform(),
    )
    test = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=mnist_transform(),
    )
    train, val = random_split(
        full_train,
        [50_000, 10_000],
        generator=torch.Generator().manual_seed(seed),
    )
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader


class DeterministicMLP(nn.Module):
    def __init__(self, hidden_units: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(784, hidden_units),
            nn.ReLU(),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.extend(
            [
                nn.Linear(hidden_units, hidden_units),
                nn.ReLU(),
            ]
        )
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_units, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def flattened_weights(self) -> np.ndarray:
        values = [
            parameter.detach().cpu().reshape(-1).numpy()
            for name, parameter in self.named_parameters()
            if name.endswith("weight")
        ]
        return np.concatenate(values)


def gaussian_log_prob_sample(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return (-0.5 * math.log(2.0 * math.pi) - torch.log(sigma) - (x - mu).pow(2) / (2.0 * sigma.pow(2))).sum()


def gaussian_log_prior(x: torch.Tensor, sigma: float) -> torch.Tensor:
    sigma_tensor = torch.as_tensor(sigma, dtype=x.dtype, device=x.device)
    return (-0.5 * math.log(2.0 * math.pi) - torch.log(sigma_tensor) - x.pow(2) / (2.0 * sigma_tensor.pow(2))).sum()


def scale_mixture_log_prior(x: torch.Tensor, pi: float, sigma1: float, sigma2: float) -> torch.Tensor:
    sigma1_tensor = torch.as_tensor(sigma1, dtype=x.dtype, device=x.device)
    sigma2_tensor = torch.as_tensor(sigma2, dtype=x.dtype, device=x.device)
    log_prob1 = -0.5 * math.log(2.0 * math.pi) - torch.log(sigma1_tensor) - x.pow(2) / (
        2.0 * sigma1_tensor.pow(2)
    )
    log_prob2 = -0.5 * math.log(2.0 * math.pi) - torch.log(sigma2_tensor) - x.pow(2) / (
        2.0 * sigma2_tensor.pow(2)
    )
    stacked = torch.stack(
        [
            math.log(pi) + log_prob1,
            math.log(1.0 - pi) + log_prob2,
        ],
        dim=0,
    )
    return torch.logsumexp(stacked, dim=0).sum()


class BayesianLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior: str,
        gaussian_prior_sigma: float,
        mixture_pi: float,
        mixture_sigma1: float,
        mixture_sigma2: float,
        mu_init_std: float = 0.1,
        rho_init_mean: float = -5.0,
        rho_init_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.prior = prior
        self.gaussian_prior_sigma = gaussian_prior_sigma
        self.mixture_pi = mixture_pi
        self.mixture_sigma1 = mixture_sigma1
        self.mixture_sigma2 = mixture_sigma2

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_rho = nn.Parameter(torch.empty(out_features))
        self.register_buffer("weight_mask", torch.ones(out_features, in_features))

        nn.init.normal_(self.weight_mu, mean=0.0, std=mu_init_std)
        nn.init.normal_(self.bias_mu, mean=0.0, std=mu_init_std)
        nn.init.normal_(self.weight_rho, mean=rho_init_mean, std=rho_init_std)
        nn.init.normal_(self.bias_rho, mean=rho_init_mean, std=rho_init_std)

        self.log_q = torch.tensor(0.0)
        self.log_p = torch.tensor(0.0)

    @staticmethod
    def sample_parameter(mu: torch.Tensor, rho: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sigma = F.softplus(rho)
        epsilon = torch.randn_like(sigma)
        return mu + sigma * epsilon, sigma

    def log_prior(self, x: torch.Tensor) -> torch.Tensor:
        if self.prior == "gaussian":
            return gaussian_log_prior(x, self.gaussian_prior_sigma)
        if self.prior == "scale_mixture":
            return scale_mixture_log_prior(x, self.mixture_pi, self.mixture_sigma1, self.mixture_sigma2)
        raise ValueError(f"Unknown prior: {self.prior}")

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if sample:
            weight, weight_sigma = self.sample_parameter(self.weight_mu, self.weight_rho)
            bias, bias_sigma = self.sample_parameter(self.bias_mu, self.bias_rho)
            self.log_q = gaussian_log_prob_sample(weight, self.weight_mu, weight_sigma)
            self.log_q = self.log_q + gaussian_log_prob_sample(bias, self.bias_mu, bias_sigma)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
            self.log_q = torch.zeros((), device=x.device, dtype=x.dtype)

        weight = weight * self.weight_mask
        self.log_p = self.log_prior(weight) + self.log_prior(bias)
        return F.linear(x, weight, bias)

    def reset_mask(self) -> None:
        self.weight_mask.fill_(1.0)

    def weight_snr(self) -> torch.Tensor:
        return self.weight_mu.detach().abs() / (F.softplus(self.weight_rho.detach()) + 1e-12)


class BayesianMLP(nn.Module):
    def __init__(
        self,
        hidden_units: int,
        prior: str,
        gaussian_prior_sigma: float,
        mixture_pi: float = 0.5,
        mixture_sigma1: float = 1.0,
        mixture_sigma2: float = math.exp(-6.0),
    ) -> None:
        super().__init__()
        kwargs = {
            "prior": prior,
            "gaussian_prior_sigma": gaussian_prior_sigma,
            "mixture_pi": mixture_pi,
            "mixture_sigma1": mixture_sigma1,
            "mixture_sigma2": mixture_sigma2,
        }
        self.flatten = nn.Flatten()
        self.fc1 = BayesianLinear(784, hidden_units, **kwargs)
        self.fc2 = BayesianLinear(hidden_units, hidden_units, **kwargs)
        self.fc3 = BayesianLinear(hidden_units, 10, **kwargs)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        x = self.flatten(x)
        x = F.relu(self.fc1(x, sample=sample))
        x = F.relu(self.fc2(x, sample=sample))
        return self.fc3(x, sample=sample)

    def kl(self) -> torch.Tensor:
        log_q = self.fc1.log_q + self.fc2.log_q + self.fc3.log_q
        log_p = self.fc1.log_p + self.fc2.log_p + self.fc3.log_p
        return log_q - log_p

    def flattened_weights(self) -> np.ndarray:
        values = [
            self.fc1.weight_mu.detach().cpu().reshape(-1).numpy(),
            self.fc2.weight_mu.detach().cpu().reshape(-1).numpy(),
            self.fc3.weight_mu.detach().cpu().reshape(-1).numpy(),
        ]
        return np.concatenate(values)

    def snr_values(self) -> torch.Tensor:
        return torch.cat(
            [
                self.fc1.weight_snr().reshape(-1),
                self.fc2.weight_snr().reshape(-1),
                self.fc3.weight_snr().reshape(-1),
            ]
        )

    def reset_masks(self) -> None:
        self.fc1.reset_mask()
        self.fc2.reset_mask()
        self.fc3.reset_mask()

    def apply_snr_pruning(self, pruning_percent: float) -> int:
        self.reset_masks()
        if pruning_percent <= 0:
            return 0

        all_snr = self.snr_values().cpu()
        threshold = torch.quantile(all_snr, pruning_percent / 100.0)
        pruned = 0
        for layer in (self.fc1, self.fc2, self.fc3):
            snr = layer.weight_snr().cpu()
            mask = (snr > threshold).to(device=layer.weight_mask.device, dtype=layer.weight_mask.dtype)
            layer.weight_mask.copy_(mask)
            pruned += int(mask.numel() - mask.sum().item())
        return pruned


@dataclass(frozen=True)
class RunConfig:
    method: str
    hidden_units: int
    learning_rate: float
    prior: str = ""
    dropout: float = 0.0
    gaussian_prior_sigma: float = 1.0
    mixture_pi: float = 0.5
    mixture_sigma1: float = 1.0
    mixture_sigma2: float = math.exp(-6.0)

    @property
    def display_method(self) -> str:
        if self.method == "vanilla":
            return "Vanilla SGD"
        if self.method == "dropout":
            return "Dropout"
        if self.method == "bbb" and self.prior == "gaussian":
            return "Bayes by Backprop (Gaussian prior)"
        if self.method == "bbb" and self.prior == "scale_mixture":
            return "Bayes by Backprop (Scale-mixture prior)"
        raise ValueError(f"Unknown method config: {self}")

    @property
    def key(self) -> tuple[str, int]:
        return (self.display_method, self.hidden_units)


@dataclass
class RunResult:
    config: RunConfig
    parameter_count: int
    best_epoch: int
    validation_error: float
    test_error: float
    selected_mc_samples: int | None
    state_dict: dict[str, torch.Tensor]
    history: dict[str, list[float]] = field(default_factory=dict)


def build_model(config: RunConfig) -> nn.Module:
    if config.method == "vanilla":
        return DeterministicMLP(config.hidden_units, dropout=0.0)
    if config.method == "dropout":
        return DeterministicMLP(config.hidden_units, dropout=config.dropout)
    if config.method == "bbb":
        return BayesianMLP(
            hidden_units=config.hidden_units,
            prior=config.prior,
            gaussian_prior_sigma=config.gaussian_prior_sigma,
            mixture_pi=config.mixture_pi,
            mixture_sigma1=config.mixture_sigma1,
            mixture_sigma2=config.mixture_sigma2,
        )
    raise ValueError(f"Unknown method: {config.method}")


def build_optimizer(model: nn.Module, lr: float, optimizer_name: str) -> torch.optim.Optimizer:
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def minibatch_kl_weight(batch_index: int, num_batches: int, scheme: str) -> float:
    if scheme == "blundell":
        return 2.0 ** (-batch_index) / (1.0 - 2.0 ** (-num_batches))
    if scheme == "uniform":
        return 1.0 / num_batches
    if scheme == "none":
        return 1.0
    raise ValueError(f"Unknown KL weighting scheme: {scheme}")


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: RunConfig,
    kl_scheme: str,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0
    num_batches = len(loader)

    for batch_index, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()

        if config.method == "bbb":
            logits = model(inputs, sample=True)
            nll = F.cross_entropy(logits, targets, reduction="sum")
            kl_weight = minibatch_kl_weight(batch_index, num_batches, kl_scheme)
            loss = nll + kl_weight * model.kl()
        else:
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets, reduction="sum")

        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_examples += targets.size(0)

    return total_loss / total_examples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: RunConfig,
    mc_samples: int = 1,
    deterministic_bbb: bool = False,
) -> dict[str, float]:
    model.eval()
    total_nll = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if config.method == "bbb":
            if deterministic_bbb:
                logits = model(inputs, sample=False)
                probs = F.softmax(logits, dim=1)
            else:
                probs = torch.stack(
                    [F.softmax(model(inputs, sample=True), dim=1) for _ in range(mc_samples)],
                    dim=0,
                ).mean(dim=0)
            log_probs = torch.log(probs.clamp_min(1e-12))
            total_nll += F.nll_loss(log_probs, targets, reduction="sum").item()
            predictions = probs.argmax(dim=1)
        else:
            logits = model(inputs)
            total_nll += F.cross_entropy(logits, targets, reduction="sum").item()
            predictions = logits.argmax(dim=1)

        total_correct += predictions.eq(targets).sum().item()
        total_examples += targets.size(0)

    accuracy = total_correct / total_examples
    return {
        "loss": total_nll / total_examples,
        "accuracy": accuracy,
        "error": 1.0 - accuracy,
    }


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def train_one_config(
    config: RunConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    optimizer_name: str,
    kl_scheme: str,
    val_mc_samples: int,
    test_mc_samples: tuple[int, ...],
    record_test_curve: bool,
    quiet: bool,
    seed: int,
) -> RunResult:
    set_seed(seed)
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config.learning_rate, optimizer_name)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    best_val_error = float("inf")
    best_epoch = 0
    best_state = cpu_state_dict(model)
    history: dict[str, list[float]] = {
        "epoch": [],
        "validation_error": [],
        "test_error": [],
    }

    started = time.time()
    for epoch in range(1, epochs + 1):
        train_epoch(model, train_loader, optimizer, device, config, kl_scheme)
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            config,
            mc_samples=val_mc_samples if config.method == "bbb" else 1,
        )

        history["epoch"].append(float(epoch))
        history["validation_error"].append(val_metrics["error"])

        if record_test_curve:
            test_metrics = evaluate(
                model,
                test_loader,
                device,
                config,
                mc_samples=max(test_mc_samples) if config.method == "bbb" else 1,
            )
            history["test_error"].append(test_metrics["error"])

        if val_metrics["error"] < best_val_error:
            best_val_error = val_metrics["error"]
            best_epoch = epoch
            best_state = cpu_state_dict(model)

        if not quiet:
            elapsed = time.time() - started
            print(
                f"{config.display_method} H={config.hidden_units} lr={config.learning_rate:g} "
                f"epoch={epoch:03d}/{epochs} val_error={100.0 * val_metrics['error']:.3f}% "
                f"elapsed={elapsed / 60.0:.1f}m",
                flush=True,
            )

    model.load_state_dict(best_state)

    selected_mc = None
    best_selection_val = float("inf")
    best_selection_test = float("inf")
    if config.method == "bbb":
        for samples in test_mc_samples:
            val_metrics = evaluate(model, val_loader, device, config, mc_samples=samples)
            test_metrics = evaluate(model, test_loader, device, config, mc_samples=samples)
            if val_metrics["error"] < best_selection_val:
                selected_mc = samples
                best_selection_val = val_metrics["error"]
                best_selection_test = test_metrics["error"]
    else:
        val_metrics = evaluate(model, val_loader, device, config)
        test_metrics = evaluate(model, test_loader, device, config)
        best_selection_val = val_metrics["error"]
        best_selection_test = test_metrics["error"]

    return RunResult(
        config=config,
        parameter_count=parameter_count,
        best_epoch=best_epoch,
        validation_error=best_selection_val,
        test_error=best_selection_test,
        selected_mc_samples=selected_mc,
        state_dict=best_state,
        history=history,
    )


def worker_train_configs(
    worker_index: int,
    device_name: str,
    configs: list[RunConfig],
    args: argparse.Namespace,
) -> list[RunResult]:
    device = get_device(device_name)
    train_loader, val_loader, test_loader = build_data_loaders(
        args.data_dir,
        args.batch_size,
        args.seed,
        args.num_workers,
        pin_memory=device.type == "cuda",
    )
    results = []
    for index, config in enumerate(configs, start=1):
        if not args.quiet:
            print(
                f"[worker {worker_index} {device}] run {index}/{len(configs)}: "
                f"{config.display_method}, H={config.hidden_units}, lr={config.learning_rate:g}",
                flush=True,
            )
        results.append(
            train_one_config(
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                epochs=args.epochs,
                optimizer_name=args.optimizer,
                kl_scheme=args.kl_weighting,
                val_mc_samples=args.val_mc_samples,
                test_mc_samples=TEST_MC_SAMPLES,
                record_test_curve=config.hidden_units == 1200,
                quiet=args.quiet,
                seed=config_seed(args.seed, config),
            )
        )
    return results


def make_run_configs(args: argparse.Namespace) -> list[RunConfig]:
    configs: list[RunConfig] = []
    hidden_units = tuple(args.hidden_units)
    learning_rates = tuple(args.learning_rates)

    if "vanilla" in args.methods:
        for hidden, lr in itertools.product(hidden_units, learning_rates):
            configs.append(RunConfig(method="vanilla", hidden_units=hidden, learning_rate=lr))

    if "dropout" in args.methods:
        for hidden, lr in itertools.product(hidden_units, learning_rates):
            configs.append(
                RunConfig(
                    method="dropout",
                    hidden_units=hidden,
                    learning_rate=lr,
                    dropout=args.dropout,
                )
            )

    if "bbb_gaussian" in args.methods:
        for hidden, lr in itertools.product(hidden_units, learning_rates):
            configs.append(
                RunConfig(
                    method="bbb",
                    hidden_units=hidden,
                    learning_rate=lr,
                    prior="gaussian",
                    gaussian_prior_sigma=args.gaussian_prior_sigma,
                )
            )

    if "bbb_scale_mixture" in args.methods:
        for hidden, lr, pi, neg_log_sigma1, neg_log_sigma2 in itertools.product(
            hidden_units,
            learning_rates,
            SCALE_MIXTURE_PI,
            SCALE_MIXTURE_NEG_LOG_SIGMA1,
            SCALE_MIXTURE_NEG_LOG_SIGMA2,
        ):
            configs.append(
                RunConfig(
                    method="bbb",
                    hidden_units=hidden,
                    learning_rate=lr,
                    prior="scale_mixture",
                    mixture_pi=pi,
                    mixture_sigma1=math.exp(-float(neg_log_sigma1)),
                    mixture_sigma2=math.exp(-float(neg_log_sigma2)),
                    gaussian_prior_sigma=args.gaussian_prior_sigma,
                )
            )

    return configs


def write_results_table(best_results: dict[tuple[str, int], RunResult], output_dir: Path) -> None:
    rows = []
    ordered_methods = [
        "Vanilla SGD",
        "Dropout",
        "Bayes by Backprop (Gaussian prior)",
        "Bayes by Backprop (Scale-mixture prior)",
    ]
    for method in ordered_methods:
        for hidden in HIDDEN_UNITS:
            result = best_results.get((method, hidden))
            if result is None:
                continue
            rows.append(
                {
                    "method": method,
                    "hidden_units": hidden,
                    "parameter_count": result.parameter_count,
                    "validation_error": f"{100.0 * result.validation_error:.4f}",
                    "test_error": f"{100.0 * result.test_error:.4f}",
                }
            )

    with (output_dir / "results_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "hidden_units", "parameter_count", "validation_error", "test_error"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_learning_curves(curve_results: dict[str, RunResult], output_dir: Path) -> None:
    plt.figure(figsize=(7.2, 4.8))
    for label, result in curve_results.items():
        epochs = result.history.get("epoch", [])
        errors = result.history.get("test_error", [])
        if not epochs or not errors:
            continue
        plt.plot(epochs, [100.0 * value for value in errors], label=label, linewidth=2.0)
    plt.xlabel("Epoch")
    plt.ylabel("Test error (%)")
    plt.title("MNIST test error vs epoch, H=1200")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curves.png", dpi=200)
    plt.close()


def load_result_model(result: RunResult, device: torch.device) -> nn.Module:
    model = build_model(result.config).to(device)
    model.load_state_dict(result.state_dict)
    model.eval()
    return model


def plot_weight_histograms(best_h1200: dict[str, RunResult], output_dir: Path, device: torch.device) -> None:
    labels = ["Bayes by Backprop", "Dropout", "Vanilla SGD"]
    result_lookup = {
        "Bayes by Backprop": best_h1200.get("Bayes by Backprop (Scale-mixture prior)")
        or best_h1200.get("Bayes by Backprop (Gaussian prior)"),
        "Dropout": best_h1200.get("Dropout"),
        "Vanilla SGD": best_h1200.get("Vanilla SGD"),
    }
    available = [(label, result) for label, result in result_lookup.items() if result is not None]
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(4.0 * len(available), 3.6), squeeze=False)
    for axis, (label, result) in zip(axes[0], available):
        model = load_result_model(result, device)
        weights = model.flattened_weights()
        axis.hist(weights, bins=120, density=True, color="#4c78a8", alpha=0.85)
        axis.set_title(label)
        axis.set_xlabel("Weight value")
        axis.set_ylabel("Density")
    fig.suptitle("Trained MNIST weight distributions, H=1200")
    fig.tight_layout()
    fig.savefig(output_dir / "weight_histograms.png", dpi=200)
    plt.close(fig)


def write_pruning_results(
    result: RunResult,
    val_loader: DataLoader,
    test_loader: DataLoader,
    output_dir: Path,
    device: torch.device,
) -> None:
    model = load_result_model(result, device)
    if not isinstance(model, BayesianMLP):
        return

    total_weights = sum(layer.weight_mask.numel() for layer in (model.fc1, model.fc2, model.fc3))
    rows = []
    mc_samples = result.selected_mc_samples or max(TEST_MC_SAMPLES)

    for percent in PRUNING_PERCENTAGES:
        pruned = model.apply_snr_pruning(float(percent))
        val_metrics = evaluate(model, val_loader, device, result.config, mc_samples=mc_samples)
        test_metrics = evaluate(model, test_loader, device, result.config, mc_samples=mc_samples)
        rows.append(
            {
                "pruning_percent": percent,
                "remaining_weights": total_weights - pruned,
                "validation_error": f"{100.0 * val_metrics['error']:.4f}",
                "test_error": f"{100.0 * test_metrics['error']:.4f}",
                "mc_samples": mc_samples,
            }
        )

    with (output_dir / "pruning_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pruning_percent",
                "remaining_weights",
                "validation_error",
                "test_error",
                "mc_samples",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    snr = model.snr_values().detach().cpu().numpy()
    plt.figure(figsize=(6.4, 4.4))
    plt.hist(np.log10(np.maximum(snr, 1e-12)), bins=120, color="#59a14f", alpha=0.9)
    plt.xlabel("log10 signal-to-noise ratio")
    plt.ylabel("Weight count")
    plt.title("Bayes by Backprop H=1200 SNR distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "snr_distribution.png", dpi=200)
    plt.close()


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    best_results: dict[tuple[str, int], RunResult],
    ran_pruning: bool,
) -> None:
    lines = [
        "# MNIST Paper Reproduction Summary",
        "",
        "This run follows the MNIST setup requested for Blundell et al., Weight Uncertainty in Neural Networks:",
        "",
        "- MNIST images are 28x28 and flattened to 784 inputs.",
        "- Pixel values are divided by 126.",
        "- The split is 50,000 train, 10,000 validation, and 10,000 test examples.",
        "- Every model uses Linear(784, H), ReLU, Linear(H, H), ReLU, Linear(H, 10).",
        f"- Hidden widths swept: {list(args.hidden_units)}.",
        f"- Batch size: {args.batch_size}. Epochs: {args.epochs}.",
        f"- Learning rates swept: {list(args.learning_rates)}.",
        f"- Optimizer: {args.optimizer}.",
        f"- Bayes by Backprop uses diagonal Gaussian posteriors with w = mu + softplus(rho) * epsilon.",
        f"- Minibatch KL weighting: {args.kl_weighting}.",
        "",
        "The result table reports validation and test errors as percentages. For Bayes by Backprop, the reported row is",
        "selected by validation error after evaluating MC sample counts in {1, 2, 5, 10}. Parameter counts are trainable",
        "parameters, so Bayes by Backprop counts both mu and rho for each weight and bias.",
        "",
        "Generated paper-style artifacts:",
        "",
        "- results_table.csv",
        "- learning_curves.png",
        "- weight_histograms.png",
        "- pruning_results.csv" if ran_pruning else "- pruning_results.csv was not generated because no H=1200 BBB result was available.",
        "- snr_distribution.png" if ran_pruning else "- snr_distribution.png was not generated because pruning was not run.",
        "",
        "Main differences from the paper are the explicit random seed, the exact software stack, and hardware-dependent",
        "floating point behavior. The Gaussian prior standard deviation is configurable; this run used",
        f"sigma={args.gaussian_prior_sigma:g}. Dropout used p={args.dropout:g}.",
        "",
        "Best selected configurations:",
        "",
    ]

    for key in sorted(best_results):
        result = best_results[key]
        config = result.config
        detail = f"- {config.display_method}, H={config.hidden_units}: lr={config.learning_rate:g}"
        if config.method == "bbb" and config.prior == "scale_mixture":
            detail += (
                f", pi={config.mixture_pi:g}, sigma1={config.mixture_sigma1:g}, "
                f"sigma2={config.mixture_sigma2:g}"
            )
        if result.selected_mc_samples is not None:
            detail += f", MC samples={result.selected_mc_samples}"
        detail += (
            f", best epoch={result.best_epoch}, validation error={100.0 * result.validation_error:.4f}%, "
            f"test error={100.0 * result.test_error:.4f}%"
        )
        lines.append(detail)

    (output_dir / "reproduction_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_curve_results(best_results: dict[tuple[str, int], RunResult]) -> dict[str, RunResult]:
    curves: dict[str, RunResult] = {}
    vanilla = best_results.get(("Vanilla SGD", 1200))
    dropout = best_results.get(("Dropout", 1200))
    bbb = best_results.get(("Bayes by Backprop (Scale-mixture prior)", 1200)) or best_results.get(
        ("Bayes by Backprop (Gaussian prior)", 1200)
    )
    if bbb is not None:
        curves["Bayes by Backprop"] = bbb
    if dropout is not None:
        curves["Dropout"] = dropout
    if vanilla is not None:
        curves["Vanilla SGD"] = vanilla
    return curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Faithful MNIST reproduction for Blundell et al.")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-units", type=int, nargs="+", default=list(HIDDEN_UNITS))
    parser.add_argument("--learning-rates", type=float, nargs="+", default=list(LEARNING_RATES))
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["vanilla", "dropout", "bbb_gaussian", "bbb_scale_mixture"],
        default=["vanilla", "dropout", "bbb_gaussian", "bbb_scale_mixture"],
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--gaussian-prior-sigma", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="sgd")
    parser.add_argument("--kl-weighting", choices=["blundell", "uniform", "none"], default="blundell")
    parser.add_argument("--val-mc-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--worker-devices",
        nargs="+",
        default=None,
        help="Run configs in parallel, one worker per listed device, e.g. cuda:0 cuda:0 cuda:1.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=Path("faithful_paper_reproduction/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("faithful_paper_reproduction/reproduction"))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    configs = make_run_configs(args)
    best_results: dict[tuple[str, int], RunResult] = {}

    if args.worker_devices:
        if not args.quiet:
            print(
                "Using parallel workers: "
                + ", ".join(f"worker {index}:{device}" for index, device in enumerate(args.worker_devices)),
                flush=True,
            )
        chunks = [configs[index :: len(args.worker_devices)] for index in range(len(args.worker_devices))]
        with mp.get_context("spawn").Pool(processes=len(args.worker_devices)) as pool:
            worker_outputs = pool.starmap(
                worker_train_configs,
                [
                    (worker_index, device_name, chunk, args)
                    for worker_index, (device_name, chunk) in enumerate(zip(args.worker_devices, chunks))
                    if chunk
                ],
            )
        for worker_results in worker_outputs:
            for result in worker_results:
                current = best_results.get(result.config.key)
                if current is None or result.validation_error < current.validation_error:
                    best_results[result.config.key] = result
    else:
        device = get_device(args.device)
        if not args.quiet:
            print(f"Using device: {device}", flush=True)

        train_loader, val_loader, test_loader = build_data_loaders(
            args.data_dir,
            args.batch_size,
            args.seed,
            args.num_workers,
            pin_memory=device.type == "cuda",
        )

        for index, config in enumerate(configs, start=1):
            record_test_curve = config.hidden_units == 1200
            if not args.quiet:
                print(
                    f"\nRun {index}/{len(configs)}: {config.display_method}, H={config.hidden_units}, "
                    f"lr={config.learning_rate:g}",
                    flush=True,
                )
            result = train_one_config(
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                epochs=args.epochs,
                optimizer_name=args.optimizer,
                kl_scheme=args.kl_weighting,
                val_mc_samples=args.val_mc_samples,
                test_mc_samples=TEST_MC_SAMPLES,
                record_test_curve=record_test_curve,
                quiet=args.quiet,
                seed=config_seed(args.seed, config),
            )
            current = best_results.get(config.key)
            if current is None or result.validation_error < current.validation_error:
                best_results[config.key] = result

    artifact_device = get_device(args.device)

    write_results_table(best_results, output_dir)
    curve_results = select_curve_results(best_results)
    plot_learning_curves(curve_results, output_dir)

    best_h1200 = {
        method: result
        for (method, hidden), result in best_results.items()
        if hidden == 1200
    }
    plot_weight_histograms(best_h1200, output_dir, artifact_device)

    pruning_source = best_results.get(("Bayes by Backprop (Scale-mixture prior)", 1200)) or best_results.get(
        ("Bayes by Backprop (Gaussian prior)", 1200)
    )
    ran_pruning = pruning_source is not None
    if pruning_source is not None:
        _, val_loader, test_loader = build_data_loaders(
            args.data_dir,
            args.batch_size,
            args.seed,
            args.num_workers,
            pin_memory=artifact_device.type == "cuda",
        )
        write_pruning_results(pruning_source, val_loader, test_loader, output_dir, artifact_device)

    write_summary(output_dir, args, best_results, ran_pruning)

    if not args.quiet:
        print(f"\nSaved reproduction artifacts in {output_dir}", flush=True)


if __name__ == "__main__":
    main()
