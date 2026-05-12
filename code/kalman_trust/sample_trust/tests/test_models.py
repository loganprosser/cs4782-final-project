from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import MODEL_NAMES, build_model


def test_all_models_return_logits() -> None:
    inputs = torch.randn(2, 1, 28, 28)
    for model_name in MODEL_NAMES:
        model = build_model(model_name, hidden_dim=64)
        model.eval()
        logits = model(inputs)
        assert logits.shape == (2, 10), model_name


def test_all_models_return_features() -> None:
    inputs = torch.randn(2, 1, 28, 28)
    for model_name in MODEL_NAMES:
        model = build_model(model_name, hidden_dim=64)
        model.eval()
        logits, features = model(inputs, return_features=True)
        assert logits.shape == (2, 10), model_name
        assert features.ndim == 2, model_name
        assert features.shape[0] == 2, model_name


if __name__ == "__main__":
    tests = [
        value
        for name, value in sorted(globals().items())
        if name.startswith("test_") and callable(value)
    ]
    for test in tests:
        test()
    print(f"passed {len(tests)} model tests")
