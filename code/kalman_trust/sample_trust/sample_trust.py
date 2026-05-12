from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


FEATURE_TRUST_MODES = {
    "final_layer_align",
    "final_layer_align_mag",
    "ema_final_layer_align",
    "combined_simple",
}

TRUST_MODES = {
    "none",
    "confidence",
    "entropy",
    "inverse_loss",
    *FEATURE_TRUST_MODES,
}


class SampleTrustState:
    def __init__(self, beta: float = 0.95) -> None:
        self.beta = beta
        self.ref_grad: Tensor | None = None
        self.step = 0


def needs_features(mode: str) -> bool:
    return mode in FEATURE_TRUST_MODES


def _ones_trust(logits: Tensor, mode: str, reason: str | None = None) -> tuple[Tensor, dict[str, Any]]:
    if reason:
        warnings.warn(f"Falling back to all-ones sample trust for mode={mode}: {reason}", RuntimeWarning)
    trust = torch.ones(logits.shape[0], device=logits.device, dtype=logits.dtype)
    return trust, _diagnostics(trust, mode)


def _diagnostics(
    trust: Tensor,
    mode: str,
    *,
    min_trust: float | None = None,
    max_trust: float | None = None,
    probs: Tensor | None = None,
    targets: Tensor | None = None,
    entropy: Tensor | None = None,
    per_example_loss: Tensor | None = None,
    alignment: Tensor | None = None,
    grad_norm: Tensor | None = None,
) -> dict[str, Any]:
    trust_detached = trust.detach()
    diag: dict[str, Any] = {
        "mode": mode,
        "trust_mean": float(trust_detached.mean().item()),
        "trust_std": float(trust_detached.std(unbiased=False).item()),
        "trust_min": float(trust_detached.min().item()),
        "trust_max": float(trust_detached.max().item()),
        "frac_at_min_trust": 0.0,
        "frac_at_max_trust": 0.0,
    }
    if min_trust is not None:
        diag["frac_at_min_trust"] = float((trust_detached <= min_trust + 1e-7).float().mean().item())
    if max_trust is not None:
        diag["frac_at_max_trust"] = float((trust_detached >= max_trust - 1e-7).float().mean().item())
    if probs is not None and targets is not None:
        confidence = probs.detach().gather(1, targets.detach().view(-1, 1)).squeeze(1)
        diag["mean_confidence"] = float(confidence.mean().item())
    if entropy is not None:
        diag["mean_entropy"] = float(entropy.detach().mean().item())
    if per_example_loss is not None:
        diag["mean_loss"] = float(per_example_loss.detach().mean().item())
    if alignment is not None:
        diag["mean_alignment"] = float(alignment.detach().mean().item())
    if grad_norm is not None:
        diag["mean_grad_norm"] = float(grad_norm.detach().mean().item())
    return diag


def _final_layer_gradients(logits: Tensor, targets: Tensor, features: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    logits_detached = logits.detach()
    features_detached = features.detach()
    probs = F.softmax(logits_detached, dim=1)
    y_onehot = F.one_hot(targets.detach(), num_classes=logits_detached.shape[1]).to(
        device=logits_detached.device,
        dtype=logits_detached.dtype,
    )
    delta = probs - y_onehot
    grad = delta.unsqueeze(2) * features_detached.unsqueeze(1)
    grad = grad.flatten(start_dim=1)
    return grad, probs, delta


def _cosine_to_reference(grad: Tensor, reference: Tensor, eps: float) -> Tensor:
    reference = reference.view(1, -1)
    numerator = (grad * reference).sum(dim=1)
    denominator = grad.norm(dim=1) * reference.norm(dim=1).clamp_min(eps)
    return numerator / denominator.clamp_min(eps)


def _batch_alignment(grad: Tensor, eps: float, use_leave_one_out: bool) -> Tensor:
    batch_size = grad.shape[0]
    if use_leave_one_out:
        if batch_size <= 1:
            return _cosine_to_reference(grad, grad.mean(dim=0), eps)
        leave_one_out = (grad.sum(dim=0, keepdim=True) - grad) / float(batch_size - 1)
        numerator = (grad * leave_one_out).sum(dim=1)
        denominator = grad.norm(dim=1) * leave_one_out.norm(dim=1)
        return numerator / denominator.clamp_min(eps)
    return _cosine_to_reference(grad, grad.mean(dim=0), eps)


def compute_sample_trust(
    mode: str,
    logits: Tensor,
    targets: Tensor,
    per_example_loss: Tensor | None = None,
    features: Tensor | None = None,
    state: SampleTrustState | None = None,
    min_trust: float = 0.05,
    max_trust: float = 1.0,
    eps: float = 1e-8,
    lambda_loss: float = 1.0,
    lambda_entropy: float = 1.0,
    use_leave_one_out: bool = False,
) -> tuple[Tensor, dict[str, Any]]:
    if mode not in TRUST_MODES:
        return _ones_trust(logits, mode, f"unknown trust mode {mode!r}")
    if logits.ndim != 2:
        return _ones_trust(logits, mode, "expected logits with shape [batch, num_classes]")
    if targets.ndim != 1 or targets.shape[0] != logits.shape[0]:
        return _ones_trust(logits, mode, "expected targets with shape [batch]")

    try:
        logits_detached = logits.detach()
        targets_detached = targets.detach()
        probs = F.softmax(logits_detached, dim=1)
        entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=1)
        loss_detached = per_example_loss.detach() if per_example_loss is not None else None

        if mode == "none":
            trust = torch.ones(logits.shape[0], device=logits.device, dtype=logits.dtype)
            diagnostics = _diagnostics(
                trust,
                mode,
                min_trust=min_trust,
                max_trust=max_trust,
                probs=probs,
                targets=targets_detached,
                entropy=entropy,
                per_example_loss=loss_detached,
            )
            return trust, diagnostics

        alignment: Tensor | None = None
        grad_norm: Tensor | None = None

        if mode == "confidence":
            trust = probs.gather(1, targets_detached.view(-1, 1)).squeeze(1)
        elif mode == "entropy":
            trust = torch.exp(-lambda_entropy * entropy)
        elif mode == "inverse_loss":
            if loss_detached is None:
                return _ones_trust(logits, mode, "per_example_loss is required")
            trust = torch.exp(-lambda_loss * loss_detached)
        else:
            if features is None:
                return _ones_trust(logits, mode, "features are required")
            grad, probs, _ = _final_layer_gradients(logits_detached, targets_detached, features)
            grad_norm = grad.norm(dim=1)
            batch_grad = grad.mean(dim=0)

            if mode == "ema_final_layer_align":
                if state is None:
                    return _ones_trust(logits, mode, "state is required")
                if state.ref_grad is None or state.ref_grad.shape != batch_grad.shape:
                    state.ref_grad = batch_grad.detach().clone()
                else:
                    state.ref_grad = (
                        state.beta * state.ref_grad.to(batch_grad.device)
                        + (1.0 - state.beta) * batch_grad.detach()
                    ).detach()
                state.step += 1
                alignment = _cosine_to_reference(grad, state.ref_grad, eps)
                trust = F.relu(alignment)
            elif mode == "final_layer_align":
                alignment = _batch_alignment(grad, eps, use_leave_one_out)
                trust = F.relu(alignment)
            elif mode == "final_layer_align_mag":
                alignment = _batch_alignment(grad, eps, use_leave_one_out)
                magnitude_center = grad_norm.median().detach() + eps
                mag = grad_norm / (grad_norm + magnitude_center)
                trust = F.relu(alignment) * mag
            elif mode == "combined_simple":
                if loss_detached is None:
                    return _ones_trust(logits, mode, "per_example_loss is required")
                alignment = _batch_alignment(grad, eps, use_leave_one_out)
                magnitude_center = grad_norm.median().detach() + eps
                mag = grad_norm / (grad_norm + magnitude_center)
                trust = F.relu(alignment) * mag * torch.exp(-lambda_loss * loss_detached)
            else:
                return _ones_trust(logits, mode, f"unhandled trust mode {mode!r}")

        trust = trust.detach().to(device=logits.device, dtype=logits.dtype)
        trust = trust.clamp(min=min_trust, max=max_trust)
        if not torch.isfinite(trust).all():
            return _ones_trust(logits, mode, "non-finite trust values")

        diagnostics = _diagnostics(
            trust,
            mode,
            min_trust=min_trust,
            max_trust=max_trust,
            probs=probs,
            targets=targets_detached,
            entropy=entropy,
            per_example_loss=loss_detached,
            alignment=alignment,
            grad_norm=grad_norm,
        )
        return trust, diagnostics
    except Exception as exc:
        return _ones_trust(logits, mode, str(exc))
