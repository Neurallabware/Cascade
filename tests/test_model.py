"""Smoke tests for CascadeNet model."""

import torch

from src.models.components.cascade_net import CascadeNet


def test_cascade_net_forward_shape():
    """Verify CascadeNet output shape for default parameters."""
    net = CascadeNet(windowsize=64)
    x = torch.randn(8, 64, 1)
    out = net(x)
    assert out.shape == (8, 1)


def test_cascade_net_custom_params():
    """Verify CascadeNet works with custom architecture parameters."""
    net = CascadeNet(
        filter_sizes=[15, 9, 3],
        filter_numbers=[16, 32, 64],
        dense_expansion=8,
        windowsize=128,
    )
    x = torch.randn(4, 128, 1)
    out = net(x)
    assert out.shape == (4, 1)


def test_cascade_net_gradient_flow():
    """Verify gradients flow through the network."""
    net = CascadeNet(windowsize=64)
    x = torch.randn(4, 64, 1, requires_grad=True)
    out = net(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
