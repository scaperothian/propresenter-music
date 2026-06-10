"""Tests for contrastive normalization."""

import numpy as np
import pytest
import torch


def test_fit_global_is_mean():
    from ppsync.transform import fit_global

    X = torch.arange(12.0).reshape(4, 3)
    g = fit_global(X)
    torch.testing.assert_close(g, X.mean(dim=0))


def test_apply_contrastive_unit_norm():
    from ppsync.transform import apply_contrastive

    X = torch.randn(20, 64)
    g = torch.randn(64)
    out = apply_contrastive(X, g)
    norms = out.norm(dim=1)
    torch.testing.assert_close(norms, torch.ones(20), atol=1e-5, rtol=1e-5)


def test_apply_contrastive_single_vector():
    from ppsync.transform import apply_contrastive

    x = torch.randn(64)
    g = torch.randn(64)
    out = apply_contrastive(x, g)
    assert out.shape == (64,)
    assert abs(out.norm().item() - 1.0) < 1e-5


def test_apply_contrastive_subtracts_global():
    from ppsync.transform import apply_contrastive

    # If x == g, output should be zero (before norm → norm is 0 → result is 0/0+eps)
    g = torch.ones(8) * 3.0
    X = g.unsqueeze(0).repeat(5, 1)  # all rows identical to g
    out = apply_contrastive(X, g)
    # After subtraction, all rows are zero; after normalization they remain near zero
    assert out.abs().max().item() < 1e-3
