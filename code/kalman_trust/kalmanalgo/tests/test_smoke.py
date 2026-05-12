from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_kalman_filter_handles_none_grads() -> None:
    from kalmanalgo.kalman_optimizer import KalmanGradientTrust

    layer = torch.nn.Linear(4, 2)
    optimizer = KalmanGradientTrust(layer.named_parameters(), lr=0.01, diag_mode="tensor")
    layer.weight.grad = torch.ones_like(layer.weight)
    layer.bias.grad = None

    diagnostics = optimizer.filter_gradients(layer.named_parameters())

    assert "weight" in diagnostics
    assert "bias" not in diagnostics
    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()


@pytest.mark.parametrize("method", ["baseline", "kalman"])
def test_one_epoch_mlp_smoke(method: str, tmp_path: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "kalmanalgo.train",
        "--model",
        "mlp",
        "--method",
        method,
        "--epochs",
        "1",
        "--batch-size",
        "32",
        "--hidden-size",
        "32",
        "--train-subset",
        "64",
        "--test-subset",
        "32",
        "--fake-data",
        "--no-plots",
        "--results-dir",
        str(tmp_path),
        "--run-name",
        f"smoke_{method}",
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert "method, model, final_test_acc, best_test_acc, final_test_loss, best_epoch" in completed.stdout
    assert (tmp_path / f"smoke_{method}" / "metrics" / "metrics.csv").exists()
    assert (tmp_path / f"smoke_{method}" / "summaries" / "final_summary.csv").exists()
