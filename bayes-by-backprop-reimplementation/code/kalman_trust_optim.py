from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch


OptimizerVariant = Literal[
    "adamw",
    "adamw_clip",
    "kalman_lr_controller",
    "direction_aware_trust",
    "per_unit_filter_trust",
    "combined_kalman_trust",
]


@dataclass
class ParamInfo:
    name: str
    param: torch.nn.Parameter
    unit_axis: int | None


def collect_param_infos(model: torch.nn.Module) -> list[ParamInfo]:
    module_by_prefix = dict(model.named_modules())
    infos: list[ParamInfo] = []
    for full_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        module_name, _, local_name = full_name.rpartition(".")
        module = module_by_prefix.get(module_name)
        unit_axis = None
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)):
            if local_name in {"weight", "bias"} and param.dim() >= 1:
                unit_axis = 0
        infos.append(ParamInfo(full_name, param, unit_axis))
    return infos


class KalmanTrustAdamW:
    def __init__(
        self,
        model: torch.nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        optimizer_variant: OptimizerVariant = "adamw",
        clip_grad_norm: float = 1.0,
        kalman_beta: float = 0.95,
        kalman_Q: float = 1e-4,
        kalman_P0: float = 1.0,
        kalman_R0: float = 1.0,
        trust_min: float = 0.05,
        trust_max: float = 1.0,
    ) -> None:
        self.param_infos = collect_param_infos(model)
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.optimizer_variant = optimizer_variant
        self.clip_grad_norm = clip_grad_norm
        self.kalman_beta = kalman_beta
        self.kalman_Q = kalman_Q
        self.kalman_P0 = kalman_P0
        self.kalman_R0 = kalman_R0
        self.trust_min = trust_min
        self.trust_max = trust_max
        self.state: dict[str, dict[str, torch.Tensor | int]] = {}
        self.last_trust: dict[str, torch.Tensor] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        for info in self.param_infos:
            if info.param.grad is None:
                continue
            if set_to_none:
                info.param.grad = None
            else:
                info.param.grad.zero_()

    def _ensure_state(self, info: ParamInfo, grad: torch.Tensor, trust_shape: torch.Size) -> dict[str, torch.Tensor | int]:
        state = self.state.setdefault(info.name, {})
        if "step" not in state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(info.param)
            state["exp_avg_sq"] = torch.zeros_like(info.param)
            state["P"] = torch.full(trust_shape, self.kalman_P0, device=grad.device, dtype=grad.dtype)
            state["R"] = torch.full(trust_shape, self.kalman_R0, device=grad.device, dtype=grad.dtype)
            state["gbar"] = torch.zeros(trust_shape, device=grad.device, dtype=grad.dtype)
            state["ema_grad"] = torch.zeros_like(grad)
        return state

    def _unit_norms(self, grad: torch.Tensor, info: ParamInfo) -> torch.Tensor:
        if info.unit_axis == 0 and grad.dim() >= 1:
            if grad.dim() == 1:
                return grad.detach().abs()
            return grad.detach().reshape(grad.size(0), -1).norm(dim=1)
        return grad.detach().norm()

    def _broadcast_trust(self, trust: torch.Tensor, param: torch.Tensor, info: ParamInfo) -> torch.Tensor:
        if trust.dim() == 0:
            return trust
        shape = [1] * param.dim()
        shape[0] = trust.numel()
        return trust.reshape(shape)

    def _trust_shape(self, grad: torch.Tensor, info: ParamInfo) -> torch.Size:
        if self.optimizer_variant in {"per_unit_filter_trust", "combined_kalman_trust"} and info.unit_axis == 0:
            return self._unit_norms(grad, info).shape
        return torch.Size([])

    def _kalman_gain(self, state: dict[str, torch.Tensor | int], z: torch.Tensor) -> torch.Tensor:
        P = state["P"]
        R = state["R"]
        gbar = state["gbar"]
        assert isinstance(P, torch.Tensor)
        assert isinstance(R, torch.Tensor)
        assert isinstance(gbar, torch.Tensor)
        gbar_new = self.kalman_beta * gbar + (1.0 - self.kalman_beta) * z
        R_new = self.kalman_beta * R + (1.0 - self.kalman_beta) * (z - gbar_new).pow(2)
        K = P / (P + R_new + self.eps)
        K = K.clamp(self.trust_min, self.trust_max)
        P_new = (1.0 - K) * P + self.kalman_Q
        state["P"] = P_new
        state["R"] = R_new
        state["gbar"] = gbar_new
        return K

    def _direction_trust(self, state: dict[str, torch.Tensor | int], grad: torch.Tensor, info: ParamInfo) -> torch.Tensor:
        ema_grad = state["ema_grad"]
        assert isinstance(ema_grad, torch.Tensor)
        if torch.count_nonzero(ema_grad).item() == 0:
            direction = torch.ones_like(self._unit_norms(grad, info))
        elif info.unit_axis == 0 and grad.dim() >= 1:
            if grad.dim() == 1:
                direction = torch.sign(grad.detach() * ema_grad).clamp(0.0, 1.0)
            else:
                g_flat = grad.detach().reshape(grad.size(0), -1)
                e_flat = ema_grad.reshape(grad.size(0), -1)
                direction = torch.nn.functional.cosine_similarity(g_flat, e_flat, dim=1).clamp(0.0, 1.0)
        else:
            direction = torch.nn.functional.cosine_similarity(
                grad.detach().reshape(1, -1),
                ema_grad.reshape(1, -1),
                dim=1,
            ).squeeze(0).clamp(0.0, 1.0)
        state["ema_grad"] = self.kalman_beta * ema_grad + (1.0 - self.kalman_beta) * grad.detach()
        return direction

    def _tensor_direction_trust(self, state: dict[str, torch.Tensor | int], grad: torch.Tensor) -> torch.Tensor:
        ema_grad = state["ema_grad"]
        assert isinstance(ema_grad, torch.Tensor)
        if torch.count_nonzero(ema_grad).item() == 0:
            direction = torch.ones((), device=grad.device, dtype=grad.dtype)
        else:
            direction = torch.nn.functional.cosine_similarity(
                grad.detach().reshape(1, -1),
                ema_grad.reshape(1, -1),
                dim=1,
            ).squeeze(0).clamp(0.0, 1.0)
        state["ema_grad"] = self.kalman_beta * ema_grad + (1.0 - self.kalman_beta) * grad.detach()
        return direction

    def _global_clip(self) -> None:
        grads = [info.param.grad for info in self.param_infos if info.param.grad is not None]
        if not grads:
            return
        total_norm = torch.norm(torch.stack([grad.detach().norm() for grad in grads]))
        clip_coef = self.clip_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for grad in grads:
                grad.mul_(clip_coef)

    def _trust_for(self, info: ParamInfo, grad: torch.Tensor, state: dict[str, torch.Tensor | int]) -> torch.Tensor:
        if self.optimizer_variant in {"adamw", "adamw_clip"}:
            return torch.ones((), device=grad.device, dtype=grad.dtype)

        tensor_norm = grad.detach().norm()
        unit_norms = self._unit_norms(grad, info)
        use_unit = self.optimizer_variant in {"per_unit_filter_trust", "combined_kalman_trust"} and info.unit_axis == 0
        z = unit_norms if use_unit else tensor_norm

        K = self._kalman_gain(state, z)
        if self.optimizer_variant == "kalman_lr_controller":
            return K
        if self.optimizer_variant == "per_unit_filter_trust":
            return K

        if self.optimizer_variant == "direction_aware_trust":
            direction = self._tensor_direction_trust(state, grad)
            return (K * direction).clamp(self.trust_min, self.trust_max)
        if self.optimizer_variant == "combined_kalman_trust":
            direction = self._direction_trust(state, grad, info)
            if not use_unit and direction.dim() > 0:
                direction = direction.mean()
            return (K * direction).clamp(self.trust_min, self.trust_max)
        raise ValueError(f"Unsupported optimizer_variant: {self.optimizer_variant}")

    @torch.no_grad()
    def step(self) -> dict[str, float]:
        if self.optimizer_variant == "adamw_clip":
            self._global_clip()

        trust_means: dict[str, float] = {}
        for info in self.param_infos:
            param = info.param
            grad = param.grad
            if grad is None:
                continue
            trust_shape = self._trust_shape(grad, info)
            state = self._ensure_state(info, grad, trust_shape)
            state["step"] = int(state["step"]) + 1
            step = int(state["step"])

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            assert isinstance(exp_avg, torch.Tensor)
            assert isinstance(exp_avg_sq, torch.Tensor)
            exp_avg.mul_(self.beta1).add_(grad, alpha=1.0 - self.beta1)
            exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)

            bias_correction1 = 1.0 - self.beta1**step
            bias_correction2 = 1.0 - self.beta2**step
            step_size = self.lr * math.sqrt(bias_correction2) / bias_correction1

            trust = self._trust_for(info, grad, state)
            trust_broadcast = self._broadcast_trust(trust, param, info)
            if self.weight_decay != 0.0:
                param.add_(param * trust_broadcast, alpha=-self.lr * self.weight_decay)
            param.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(self.eps), value=-step_size)
            if trust_broadcast.dim() == 0:
                # addcdiv cannot directly take a tensor-valued value argument, so correct
                # the just-applied full AdamW step when trust is not scalar-one.
                if float(trust_broadcast.item()) != 1.0:
                    raw_delta = exp_avg / exp_avg_sq.sqrt().add(self.eps)
                    param.add_(raw_delta, alpha=step_size * (1.0 - float(trust_broadcast.item())))
            else:
                raw_delta = exp_avg / exp_avg_sq.sqrt().add(self.eps)
                param.add_(raw_delta * (1.0 - trust_broadcast), alpha=step_size)

            self.last_trust[info.name] = trust.detach().float().cpu()
            trust_means[info.name] = float(trust.detach().float().mean().item())
        return trust_means

    def trust_snapshot(self) -> dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in self.last_trust.items()}
