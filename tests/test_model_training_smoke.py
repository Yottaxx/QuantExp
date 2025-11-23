import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")

import torch
import numpy as np

from src.model import PatchTSTForStock, SotaConfig, HybridLoss


def test_model_forward_and_loss():
    config = SotaConfig(
        num_input_channels=3,
        context_length=5,
        patch_length=2,
        stride=1,
        d_model=16,
        num_hidden_layers=1,
        n_heads=2,
        dropout=0.1,
        mse_weight=0.4,
    )
    model = PatchTSTForStock(config)

    batch = torch.tensor(np.random.randn(4, config.num_input_channels, config.context_length), dtype=torch.float32)
    labels = torch.tensor(np.random.randn(4, 1), dtype=torch.float32)

    out = model(past_values=batch, labels=labels)
    assert out.logits.shape[0] == batch.shape[0]
    assert out.loss is not None
    assert torch.isfinite(out.loss).item()

    # Explicitly check HybridLoss behavior on simple inputs
    loss_fn = HybridLoss(mse_weight=0.5)
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.5, 2.5, 3.5])
    loss = loss_fn(pred, target)
    assert loss > 0
