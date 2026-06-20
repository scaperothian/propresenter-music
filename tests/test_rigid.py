"""Tests for the rigid (no-warp) matcher."""

import numpy as np

from ppsync.rigid import rigid_align


def _unit_rows(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def test_rigid_align_finds_exact_subsequence():
    rng = np.random.default_rng(7)
    ref = _unit_rows(rng.normal(size=(400, 16)).astype(np.float32))
    ts = np.arange(400) * 0.05
    # live = every 4th ref frame ending at index 215 (t = 10.75)
    step = 4
    end = 215
    m = 20
    live = ref[end - (m - 1) * step:end + 1:step]
    r = rigid_align(live, ref, ts, 0.0, 20.0, live_step=step, top_k=3)
    assert r["refined_t"] == ts[end]
    assert r["confidence"] > 0.99
    assert r["cost_margin"] > 0.1  # random elsewhere — clear winner


def test_rigid_align_empty_window_returns_zero_confidence():
    ref = _unit_rows(np.ones((10, 4), dtype=np.float32))
    ts = np.arange(10) * 0.05
    live = ref[:3]
    r = rigid_align(live, ref, ts, 9.0, 9.1, live_step=4)
    assert r["confidence"] == 0.0
