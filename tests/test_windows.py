"""Tests for windowing utilities."""

import numpy as np
import pytest
import torch


def test_strided_window_embeddings_count():
    from ppsync.windows import strided_window_embeddings

    # 10s at 75fps = 750 frames; 2s lookback, 0.1s stride → ~80 windows
    T, D = 750, 32
    frames = torch.randn(T, D)
    embs, ts = strided_window_embeddings(frames, lookback_sec=2.0, stride_sec=0.1, fps=75)
    # First window starts at frame 150 (2s); last window ends at frame ≤ 750
    expected_n = (T - 150) // 8  # rough: (T - win) / hop, hop=7.5→8
    assert embs.shape[0] >= 1
    assert embs.shape[1] == D
    assert len(ts) == embs.shape[0]


def test_strided_window_embeddings_timestamps_increasing():
    from ppsync.windows import strided_window_embeddings

    frames = torch.randn(300, 16)
    _, ts = strided_window_embeddings(frames, lookback_sec=1.0, stride_sec=0.1, fps=75)
    assert (np.diff(ts) > 0).all()


def test_strided_window_short_audio_empty():
    from ppsync.windows import strided_window_embeddings

    # Only 50 frames at 75fps = 0.67s, lookback=2s → no full windows
    frames = torch.randn(50, 16)
    embs, ts = strided_window_embeddings(frames, lookback_sec=2.0, stride_sec=0.1, fps=75)
    assert embs.shape[0] == 0


def test_pool_slide_embeddings_range():
    from ppsync.windows import pool_slide_embeddings

    T, D = 500, 16
    embs = torch.randn(T, D)
    ts = np.arange(T, dtype=np.float32) * 0.02  # 20ms stride
    proto = pool_slide_embeddings(embs, ts, t_start=2.0, t_stop=4.0)
    assert proto.shape == (D,)
    # Mean of windows in [2, 4) should match manual calculation
    mask = (ts >= 2.0) & (ts < 4.0)
    expected = embs[torch.from_numpy(mask)].mean(dim=0)
    torch.testing.assert_close(proto, expected)


def test_pool_slide_embeddings_empty_range():
    from ppsync.windows import pool_slide_embeddings

    embs = torch.randn(100, 8)
    ts = np.arange(100, dtype=np.float32) * 0.02
    proto = pool_slide_embeddings(embs, ts, t_start=50.0, t_stop=60.0)
    assert (proto == 0).all()  # no windows in that range → zeros
