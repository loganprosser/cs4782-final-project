from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import torch
from torch import nn


NamedParam = tuple[str, nn.Parameter]


class KalmanGradientTrust:
    """Filter minibatch gradients with a diagonal Kalman-style trust estimate.

    This class deliberately does not sample weights and does not store weight
    distributions. The network forward pass remains deterministic. After
    ``loss.backward()``, call ``filter_gradients`` or use ``step`` directly to
    replace each raw gradient with its filtered estimate before the wrapped
    optimizer updates the parameters.
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[NamedParam],
        base_optimizer: str = "sgd",
        lr: float = 1e-2,
        P0: float = 1.0,
        Q: float = 1e-5,
        R_floor: float = 1e-8,
        beta_R: float = 0.95,
        eps: float = 1e-8,
        diag_mode: str = "tensor",
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        **optimizer_kwargs: Any,
    ) -> None:
        items = list(params)
        if not items:
            raise ValueError("KalmanGradientTrust received no parameters.")
        if isinstance(items[0], tuple):
            self.named_params = [(str(name), param) for name, param in items]  # type: ignore[misc]
        else:
            self.named_params = [(f"param_{idx}", param) for idx, param in enumerate(items)]  # type: ignore[arg-type]
        self.params = [param for _, param in self.named_params]

        if diag_mode not in {"tensor", "scalar"}:
            raise ValueError("diag_mode must be 'tensor' or 'scalar'.")
        self.P0 = float(P0)
        self.Q = float(Q)
        self.R_floor = float(R_floor)
        self.beta_R = float(beta_R)
        self.eps = float(eps)
        self.diag_mode = diag_mode
        self.kalman_state: dict[str, dict[str, torch.Tensor]] = {}
        self.last_diagnostics: dict[str, dict[str, float]] = {}

        base = base_optimizer.lower()
        if base == "sgd":
            self.base_optimizer = torch.optim.SGD(
                self.params,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                **optimizer_kwargs,
            )
        elif base == "adam":
            self.base_optimizer = torch.optim.Adam(
                self.params,
                lr=lr,
                weight_decay=weight_decay,
                **optimizer_kwargs,
            )
        else:
            raise ValueError("base_optimizer must be 'sgd' or 'adam'.")

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self.base_optimizer.param_groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Any | None = None, filter_gradients: bool = True) -> Any:
        if filter_gradients:
            self.filter_gradients()
        return self.base_optimizer.step(closure=closure)

    def _state_shape(self, param: nn.Parameter) -> tuple[int, ...]:
        return () if self.diag_mode == "scalar" else tuple(param.shape)

    def _init_state(self, key: str, param: nn.Parameter) -> dict[str, torch.Tensor]:
        shape = self._state_shape(param)
        options = {"device": param.device, "dtype": param.dtype}
        state = {
            "m": torch.zeros_like(param, memory_format=torch.preserve_format),
            "P": torch.full(shape, self.P0, **options),
            "grad_mean": torch.zeros(shape, **options),
            "grad_var": torch.zeros(shape, **options),
            "K": torch.zeros(shape, **options),
            "R": torch.full(shape, self.R_floor, **options),
        }
        self.kalman_state[key] = state
        return state

    def _get_state(self, key: str, param: nn.Parameter) -> dict[str, torch.Tensor]:
        state = self.kalman_state.get(key)
        if state is None or state["m"].device != param.device or state["m"].shape != param.shape:
            return self._init_state(key, param)
        if self.diag_mode == "tensor" and state["P"].shape != param.shape:
            return self._init_state(key, param)
        if self.diag_mode == "scalar" and state["P"].ndim != 0:
            return self._init_state(key, param)
        return state

    @torch.no_grad()
    def filter_gradients(self, named_params: Iterable[NamedParam] | None = None) -> dict[str, dict[str, float]]:
        diagnostics: dict[str, dict[str, float]] = {}
        params_iter: Iterator[NamedParam]
        params_iter = iter(self.named_params if named_params is None else named_params)

        for name, param in params_iter:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            raw_grad_norm = float(grad.norm().item())
            state = self._get_state(name, param)
            m = state["m"]

            if self.diag_mode == "tensor":
                state["grad_mean"].mul_(self.beta_R).add_(grad, alpha=1.0 - self.beta_R)
                centered = grad - state["grad_mean"]
                state["grad_var"].mul_(self.beta_R).addcmul_(centered, centered, value=1.0 - self.beta_R)
            else:
                state["grad_mean"].mul_(self.beta_R).add_(grad.mean(), alpha=1.0 - self.beta_R)
                centered = grad - state["grad_mean"]
                var_observation = centered.square().mean()
                state["grad_var"].mul_(self.beta_R).add_(var_observation, alpha=1.0 - self.beta_R)

            R = state["grad_var"].add(self.R_floor)
            P = state["P"]
            K = P / (P + R + self.eps)
            m.addcmul_(K, grad - m)
            P.mul_(1.0 - K).add_(self.Q)

            state["K"].copy_(K)
            state["R"].copy_(R)
            param.grad.copy_(m)

            diagnostics[name] = {
                "raw_grad_norm": raw_grad_norm,
                "filtered_grad_norm": float(m.norm().item()),
                "kalman_gain": float(K.mean().item()),
                "measurement_R": float(R.mean().item()),
                "covariance_P": float(P.mean().item()),
            }

        self.last_diagnostics = diagnostics
        return diagnostics

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "kalman_state": self.kalman_state,
            "settings": {
                "P0": self.P0,
                "Q": self.Q,
                "R_floor": self.R_floor,
                "beta_R": self.beta_R,
                "eps": self.eps,
                "diag_mode": self.diag_mode,
            },
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.kalman_state = state_dict.get("kalman_state", {})
