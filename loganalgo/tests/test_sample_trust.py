from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sample_trust import SampleTrustState, TRUST_MODES, compute_sample_trust


def make_batch(batch_size: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(7)
    logits = torch.randn(batch_size, 4, requires_grad=True)
    targets = torch.randint(0, 4, (batch_size,))
    features = torch.randn(batch_size, 6, requires_grad=True)
    per_example_loss = F.cross_entropy(logits, targets, reduction="none")
    return logits, targets, features, per_example_loss


def test_none_returns_all_ones() -> None:
    logits, targets, features, per_example_loss = make_batch()
    trust, _ = compute_sample_trust("none", logits, targets, per_example_loss, features)
    assert torch.allclose(trust, torch.ones_like(trust))


def test_all_modes_return_trust_shape() -> None:
    logits, targets, features, per_example_loss = make_batch()
    for mode in TRUST_MODES:
        state = SampleTrustState() if mode == "ema_final_layer_align" else None
        trust, _ = compute_sample_trust(
            mode,
            logits,
            targets,
            per_example_loss=per_example_loss,
            features=features,
            state=state,
            min_trust=0.05,
            max_trust=1.0,
        )
        assert trust.shape == targets.shape


def test_all_modes_return_finite_trust() -> None:
    logits, targets, features, per_example_loss = make_batch()
    for mode in TRUST_MODES:
        state = SampleTrustState() if mode == "ema_final_layer_align" else None
        trust, _ = compute_sample_trust(
            mode,
            logits,
            targets,
            per_example_loss=per_example_loss,
            features=features,
            state=state,
            min_trust=0.05,
            max_trust=1.0,
        )
        assert torch.isfinite(trust).all()


def test_all_modes_return_trust_within_bounds() -> None:
    logits, targets, features, per_example_loss = make_batch()
    for mode in TRUST_MODES:
        state = SampleTrustState() if mode == "ema_final_layer_align" else None
        trust, _ = compute_sample_trust(
            mode,
            logits,
            targets,
            per_example_loss=per_example_loss,
            features=features,
            state=state,
            min_trust=0.05,
            max_trust=1.0,
        )
        assert float(trust.min()) >= 0.05 - 1e-7
        assert float(trust.max()) <= 1.0 + 1e-7


def test_final_layer_align_modes_work_with_features() -> None:
    logits, targets, features, per_example_loss = make_batch()
    for mode in ["final_layer_align", "final_layer_align_mag", "combined_simple"]:
        trust, diag = compute_sample_trust(mode, logits, targets, per_example_loss, features=features)
        assert trust.shape == targets.shape
        assert torch.isfinite(trust).all()
        assert "mean_alignment" in diag
        assert "mean_grad_norm" in diag


def test_final_layer_align_modes_fall_back_without_features() -> None:
    logits, targets, _, per_example_loss = make_batch()
    for mode in ["final_layer_align", "final_layer_align_mag", "ema_final_layer_align", "combined_simple"]:
        trust, _ = compute_sample_trust(mode, logits, targets, per_example_loss, features=None)
        assert torch.allclose(trust, torch.ones_like(trust))


def test_ema_final_layer_align_updates_state() -> None:
    logits, targets, features, per_example_loss = make_batch()
    state = SampleTrustState(beta=0.9)
    trust, _ = compute_sample_trust(
        "ema_final_layer_align",
        logits,
        targets,
        per_example_loss=per_example_loss,
        features=features,
        state=state,
    )
    assert trust.shape == targets.shape
    assert state.ref_grad is not None
    assert state.step == 1


def test_weighted_loss_none_matches_mean_ce() -> None:
    logits, targets, features, per_example_loss = make_batch()
    trust, _ = compute_sample_trust("none", logits, targets, per_example_loss, features)
    weighted_loss = (trust.detach() * per_example_loss).sum() / (trust.detach().sum() + 1e-8)
    ordinary_loss = F.cross_entropy(logits, targets)
    assert torch.allclose(weighted_loss, ordinary_loss)


def test_leave_one_out_does_not_crash_for_batch_size_gt_one() -> None:
    logits, targets, features, per_example_loss = make_batch(batch_size=5)
    trust, _ = compute_sample_trust(
        "final_layer_align",
        logits,
        targets,
        per_example_loss=per_example_loss,
        features=features,
        use_leave_one_out=True,
    )
    assert trust.shape == targets.shape
    assert torch.isfinite(trust).all()


def test_leave_one_out_batch_size_one_falls_back_gracefully() -> None:
    logits, targets, features, per_example_loss = make_batch(batch_size=1)
    trust, _ = compute_sample_trust(
        "final_layer_align",
        logits,
        targets,
        per_example_loss=per_example_loss,
        features=features,
        use_leave_one_out=True,
    )
    assert trust.shape == targets.shape
    assert torch.isfinite(trust).all()


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"passed {len(tests)} sample_trust tests")
