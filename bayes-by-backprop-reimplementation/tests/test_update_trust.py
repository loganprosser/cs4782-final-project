from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from models import StandardMLP
from update_trust import (
    apply_kalman_layer_trust,
    apply_update_trust_scaling,
    build_optimizer,
    get_layer_param_groups,
)


class UpdateTrustTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = StandardMLP(hidden_dim=16)
        self.inputs = torch.randn(8, 1, 28, 28)
        self.targets = torch.randint(0, 10, (8,))

    def _backward_once(self) -> None:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(self.inputs)
        loss = torch.nn.functional.cross_entropy(logits, self.targets)
        loss.backward()

    def test_depth_decay_creates_decreasing_param_group_lrs(self) -> None:
        optimizer = build_optimizer(
            self.model,
            base_lr=1e-3,
            weight_decay=0.0,
            update_trust_mode="depth_decay",
            depth_decay_lambda=0.15,
            optimizer_cls=torch.optim.Adam,
        )
        lrs = [group["lr"] for group in optimizer.param_groups]
        self.assertGreater(len(lrs), 1)
        for earlier, later in zip(lrs, lrs[1:]):
            self.assertGreaterEqual(earlier, later)

    def test_grad_norm_scales_gradients_without_nans(self) -> None:
        self._backward_once()
        first_group = get_layer_param_groups(self.model)[0]
        param = first_group["params"][0]
        original_grad = param.grad.detach().clone()

        stats = apply_update_trust_scaling(
            model=self.model,
            trust_state={},
            mode="grad_norm",
            eps=1e-8,
            clip_min=0.1,
            clip_max=10.0,
        )
        self.assertTrue(stats)
        self.assertFalse(torch.isnan(param.grad).any())
        self.assertFalse(torch.allclose(original_grad, param.grad))

    def test_running_grad_var_updates_state_across_steps(self) -> None:
        trust_state: dict[str, float] = {}
        self._backward_once()
        apply_update_trust_scaling(
            model=self.model,
            trust_state=trust_state,
            mode="running_grad_var",
            beta=0.95,
            eps=1e-8,
            clip_min=0.1,
            clip_max=10.0,
        )
        self.assertTrue(trust_state)
        first_snapshot = trust_state.copy()

        self._backward_once()
        apply_update_trust_scaling(
            model=self.model,
            trust_state=trust_state,
            mode="running_grad_var",
            beta=0.95,
            eps=1e-8,
            clip_min=0.1,
            clip_max=10.0,
        )
        changed = any(not math.isclose(first_snapshot[key], trust_state[key]) for key in first_snapshot)
        self.assertTrue(changed)

    def test_none_mode_preserves_gradients(self) -> None:
        self._backward_once()
        gradients_before = [param.grad.detach().clone() for param in self.model.parameters() if param.grad is not None]

        apply_update_trust_scaling(
            model=self.model,
            trust_state={},
            mode="none",
            eps=1e-8,
            clip_min=0.1,
            clip_max=10.0,
        )
        gradients_after = [param.grad.detach().clone() for param in self.model.parameters() if param.grad is not None]
        for before, after in zip(gradients_before, gradients_after):
            self.assertTrue(torch.allclose(before, after))

    def test_kalman_initializes_layer_state(self) -> None:
        self._backward_once()
        trust_state: dict[str, dict[str, float]] = {}
        stats = apply_kalman_layer_trust(self.model, trust_state)
        self.assertTrue(stats)
        self.assertTrue(trust_state)
        for layer_state in trust_state.values():
            for key in ["P", "R", "gbar", "K", "step"]:
                self.assertIn(key, layer_state)

    def test_kalman_scales_gradients_by_K(self) -> None:
        self._backward_once()
        original_norms = {}
        for group in get_layer_param_groups(self.model):
            name = group["name"]
            total = 0.0
            for param in group["params"]:
                if param.grad is not None:
                    total += float(param.grad.detach().pow(2).sum().item())
            original_norms[name] = math.sqrt(total)

        trust_state: dict[str, dict[str, float]] = {}
        apply_kalman_layer_trust(self.model, trust_state)

        for group in get_layer_param_groups(self.model):
            name = group["name"]
            total = 0.0
            for param in group["params"]:
                if param.grad is not None:
                    total += float(param.grad.detach().pow(2).sum().item())
            new_norm = math.sqrt(total)
            K = trust_state[name]["K"]
            self.assertAlmostEqual(new_norm, original_norms[name] * K, places=4)

    def test_kalman_state_updates_over_two_steps(self) -> None:
        trust_state: dict[str, dict[str, float]] = {}
        self._backward_once()
        apply_kalman_layer_trust(self.model, trust_state)
        first_state = {name: values.copy() for name, values in trust_state.items()}

        self._backward_once()
        apply_kalman_layer_trust(self.model, trust_state)

        changed = False
        for name, values in trust_state.items():
            if (
                not math.isclose(first_state[name]["P"], values["P"])
                or not math.isclose(first_state[name]["R"], values["R"])
                or not math.isclose(first_state[name]["step"], values["step"])
            ):
                changed = True
        self.assertTrue(changed)

    def test_kalman_values_are_finite(self) -> None:
        self._backward_once()
        trust_state: dict[str, dict[str, float]] = {}
        apply_kalman_layer_trust(self.model, trust_state)
        for values in trust_state.values():
            self.assertTrue(math.isfinite(values["K"]))
            self.assertTrue(math.isfinite(values["P"]))
            self.assertTrue(math.isfinite(values["R"]))


if __name__ == "__main__":
    unittest.main()
