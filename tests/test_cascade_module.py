"""Smoke tests for CascadeLitModule."""

from functools import partial

import torch
from torch.optim import Adagrad

from src.models.cascade_module import CascadeLitModule
from src.models.components.cascade_net import CascadeNet


def test_lit_module_forward():
    """Verify LightningModule forward pass."""
    net = CascadeNet(windowsize=64)
    model = CascadeLitModule(
        net=net,
        optimizer=partial(Adagrad, lr=0.05),
    )
    x = torch.randn(4, 64, 1)
    out = model(x)
    assert out.shape == (4, 1)


def test_lit_module_training_step():
    """Verify training_step runs without error."""
    net = CascadeNet(windowsize=64)
    model = CascadeLitModule(
        net=net,
        optimizer=partial(Adagrad, lr=0.05),
    )
    x = torch.randn(8, 64, 1)
    y = torch.randn(8, 1)
    loss = model.training_step((x, y), batch_idx=0)
    assert loss.ndim == 0  # scalar
    assert loss.item() > 0


def test_lit_module_configure_optimizers():
    """Verify optimizer configuration."""
    net = CascadeNet(windowsize=64)
    model = CascadeLitModule(
        net=net,
        optimizer=partial(Adagrad, lr=0.05),
    )
    config = model.configure_optimizers()
    assert "optimizer" in config
    assert isinstance(config["optimizer"], Adagrad)
