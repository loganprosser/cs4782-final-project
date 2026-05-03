from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch


TRUST_MODES = {"none", "depth_decay", "grad_norm", "running_grad_var", "kalman_layer"}


def get_layer_param_groups(model: torch.nn.Module) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    seen_param_ids: set[int] = set()
    depth_index = 0

    for module_name, module in model.named_modules():
        direct_params = []
        for _, param in module.named_parameters(recurse=False):
            if not param.requires_grad or id(param) in seen_param_ids:
                continue
            seen_param_ids.add(id(param))
            direct_params.append(param)

        if direct_params:
            group_name = module_name or "root"
            groups.append({"name": group_name, "params": direct_params, "depth": depth_index})
            depth_index += 1

    if groups:
        return groups

    fallback_groups = []
    for depth_index, (param_name, param) in enumerate(model.named_parameters()):
        if not param.requires_grad or id(param) in seen_param_ids:
            continue
        fallback_groups.append({"name": param_name, "params": [param], "depth": depth_index})
    return fallback_groups


def build_optimizer(
    model: torch.nn.Module,
    base_lr: float,
    weight_decay: float,
    update_trust_mode: str,
    depth_decay_lambda: float = 0.15,
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.Adam,
    **optimizer_kwargs,
) -> torch.optim.Optimizer:
    if update_trust_mode not in TRUST_MODES:
        raise ValueError(f"Unsupported update_trust_mode: {update_trust_mode}")

    if update_trust_mode == "depth_decay":
        param_groups = []
        for group in get_layer_param_groups(model):
            depth = int(group["depth"])
            alpha = math.exp(-depth_decay_lambda * depth)
            param_groups.append(
                {
                    "params": group["params"],
                    "lr": base_lr * alpha,
                    "weight_decay": weight_decay,
                    "trust_alpha": alpha,
                    "group_name": group["name"],
                }
            )
        return optimizer_cls(param_groups, **optimizer_kwargs)

    return optimizer_cls(model.parameters(), lr=base_lr, weight_decay=weight_decay, **optimizer_kwargs)


def _iter_group_grads(params: Iterable[torch.nn.Parameter]) -> list[torch.Tensor]:
    grads = []
    for param in params:
        if not param.requires_grad or param.grad is None:
            continue
        grads.append(param.grad)
    return grads


def _compute_group_grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    grads = _iter_group_grads(params)
    if not grads:
        return 0.0
    total = 0.0
    for grad in grads:
        total += float(grad.detach().pow(2).sum().item())
    return math.sqrt(total)


def _scale_group_grads(params: Iterable[torch.nn.Parameter], alpha: float) -> None:
    for param in params:
        if not param.requires_grad or param.grad is None:
            continue
        param.grad.mul_(alpha)


def apply_kalman_layer_trust(
    model: torch.nn.Module,
    trust_state: dict[str, dict[str, float]],
    beta: float = 0.95,
    process_noise: float = 1e-4,
    initial_P: float = 1.0,
    initial_R: float = 1.0,
    eps: float = 1e-8,
    clip_min: float = 0.05,
    clip_max: float = 1.0,
) -> dict[str, float]:
    # This is not a full Extended Kalman Filter. It is a Kalman-inspired layerwise
    # adaptive gradient trust mechanism. P represents layer update uncertainty,
    # R represents estimated gradient measurement noise, and K scales the optimizer update.
    stats: dict[str, float] = {}

    for group in get_layer_param_groups(model):
        group_name = str(group["name"])
        params = group["params"]
        grad_norm = _compute_group_grad_norm(params)
        if grad_norm == 0.0:
            continue

        if group_name not in trust_state:
            initial_K = initial_P / (initial_P + initial_R + eps)
            trust_state[group_name] = {
                "P": initial_P,
                "R": initial_R,
                "gbar": grad_norm,
                "K": initial_K,
                "step": 0.0,
            }

        layer_state = trust_state[group_name]
        P_old = float(layer_state["P"])
        R_old = float(layer_state["R"])
        gbar_old = float(layer_state["gbar"])

        gbar_new = beta * gbar_old + (1.0 - beta) * grad_norm
        R_new = beta * R_old + (1.0 - beta) * ((grad_norm - gbar_new) ** 2)
        K = P_old / (P_old + R_new + eps)
        K = max(clip_min, min(clip_max, K))
        _scale_group_grads(params, K)
        P_new = (1.0 - K) * P_old + process_noise

        layer_state["P"] = P_new
        layer_state["R"] = R_new
        layer_state["gbar"] = gbar_new
        layer_state["K"] = K
        layer_state["step"] = float(layer_state["step"]) + 1.0

        stats[f"kalman/{group_name}/K"] = K
        stats[f"kalman/{group_name}/P"] = P_new
        stats[f"kalman/{group_name}/R"] = R_new
        stats[f"kalman/{group_name}/grad_norm"] = grad_norm

    return stats


def apply_update_trust_scaling(
    model: torch.nn.Module,
    trust_state: dict[str, Any],
    mode: str,
    beta: float = 0.95,
    eps: float = 1e-8,
    clip_min: float = 0.1,
    clip_max: float = 10.0,
    kalman_beta: float = 0.95,
    kalman_process_noise: float = 1e-4,
    kalman_initial_P: float = 1.0,
    kalman_initial_R: float = 1.0,
    kalman_eps: float = 1e-8,
    kalman_clip_min: float = 0.05,
    kalman_clip_max: float = 1.0,
) -> dict[str, float]:
    if mode not in TRUST_MODES:
        raise ValueError(f"Unsupported update_trust_mode: {mode}")
    if mode in {"none", "depth_decay"}:
        return {}
    if mode == "kalman_layer":
        return apply_kalman_layer_trust(
            model=model,
            trust_state=trust_state,  # type: ignore[arg-type]
            beta=kalman_beta,
            process_noise=kalman_process_noise,
            initial_P=kalman_initial_P,
            initial_R=kalman_initial_R,
            eps=kalman_eps,
            clip_min=kalman_clip_min,
            clip_max=kalman_clip_max,
        )

    stats: dict[str, float] = {}
    for group in get_layer_param_groups(model):
        group_name = str(group["name"])
        params = group["params"]
        grad_norm = _compute_group_grad_norm(params)
        if grad_norm == 0.0:
            continue

        if mode == "grad_norm":
            alpha = 1.0 / (grad_norm + eps)
        else:
            previous_var = trust_state.get(group_name, 0.0)
            running_var = beta * previous_var + (1.0 - beta) * (grad_norm ** 2)
            trust_state[group_name] = running_var
            alpha = 1.0 / math.sqrt(running_var + eps)
            stats[f"trust/{group_name}/running_var"] = running_var

        alpha = max(clip_min, min(clip_max, alpha))
        _scale_group_grads(params, alpha)
        stats[f"trust/{group_name}/alpha"] = alpha
        stats[f"trust/{group_name}/grad_norm"] = grad_norm

    return stats


def summarize_trust_logs(epoch_trust_logs: list[dict[str, float]]) -> dict[str, float]:
    alpha_values = []
    grad_norm_values = []
    running_var_values = []
    kalman_K_values = []
    kalman_P_values = []
    kalman_R_values = []

    for batch_stats in epoch_trust_logs:
        for key, value in batch_stats.items():
            if key.endswith("/alpha"):
                alpha_values.append(value)
            elif key.endswith("/grad_norm"):
                grad_norm_values.append(value)
            elif key.endswith("/running_var"):
                running_var_values.append(value)
            elif key.endswith("/K"):
                kalman_K_values.append(value)
            elif key.endswith("/P"):
                kalman_P_values.append(value)
            elif key.endswith("/R"):
                kalman_R_values.append(value)

    summary: dict[str, float] = {}
    if alpha_values:
        summary["avg_alpha"] = sum(alpha_values) / len(alpha_values)
        summary["min_alpha"] = min(alpha_values)
        summary["max_alpha"] = max(alpha_values)
    if grad_norm_values:
        summary["avg_grad_norm"] = sum(grad_norm_values) / len(grad_norm_values)
    if running_var_values:
        summary["avg_running_var"] = sum(running_var_values) / len(running_var_values)
    if kalman_K_values:
        summary["kalman_mean_K"] = sum(kalman_K_values) / len(kalman_K_values)
        summary["kalman_min_K"] = min(kalman_K_values)
        summary["kalman_max_K"] = max(kalman_K_values)
    if kalman_P_values:
        summary["kalman_mean_P"] = sum(kalman_P_values) / len(kalman_P_values)
    if kalman_R_values:
        summary["kalman_mean_R"] = sum(kalman_R_values) / len(kalman_R_values)
    return summary


def get_update_trust_suffix(
    mode: str,
    depth_decay_lambda: float,
    running_grad_beta: float,
    kalman_beta: float = 0.95,
    kalman_process_noise: float = 1e-4,
    kalman_initial_P: float = 1.0,
    kalman_initial_R: float = 1.0,
) -> str:
    if mode == "none":
        return "trust_none"
    if mode == "depth_decay":
        return f"trust_depth_lambda{depth_decay_lambda}"
    if mode == "grad_norm":
        return "trust_gradnorm"
    if mode == "running_grad_var":
        return f"trust_gradvar_beta{running_grad_beta}"
    if mode == "kalman_layer":
        return f"trust_kalmanLayer_beta{kalman_beta}_Q{kalman_process_noise}_P{kalman_initial_P}_R{kalman_initial_R}"
    raise ValueError(f"Unsupported update_trust_mode: {mode}")
