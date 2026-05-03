import torch
from torch.nn import functional as F


def bayesian_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    log_q: torch.Tensor,
    log_p: torch.Tensor,
    dataset_size: int,
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce = F.cross_entropy(logits, targets, reduction="mean")
    kl = (log_q - log_p) / dataset_size
    loss = ce + kl_weight * kl
    return loss, {"nll": ce.item(), "kl": kl.item(), "loss": loss.item()}


def bayesian_regression_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    log_q: torch.Tensor,
    log_p: torch.Tensor,
    dataset_size: int,
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    mse = F.mse_loss(preds, targets, reduction="mean")
    kl = (log_q - log_p) / dataset_size
    loss = mse + kl_weight * kl
    return loss, {"mse": mse.item(), "kl": kl.item(), "loss": loss.item()}
