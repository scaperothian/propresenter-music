"""Dense strided window embeddings over MERT frame sequences."""

from __future__ import annotations

import numpy as np
import torch

from .config import LOOKBACK_SEC, MERT_FRAME_RATE, STRIDE_SEC


def strided_window_embeddings(
    frame_embs: torch.Tensor,   # [T, D] MERT frame embeddings for one layer
    lookback_sec: float = LOOKBACK_SEC,
    stride_sec: float = STRIDE_SEC,
    fps: int = MERT_FRAME_RATE,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Build dense strided window embeddings by mean-pooling MERT frames.

    For each window position (spaced by *stride_sec*), pool all frames in
    [t - lookback_sec, t].  Positions where the lookback window extends
    before the start of the recording are skipped.

    Args:
        frame_embs:   [T, D] raw frame embeddings (un-normalized)
        lookback_sec: window duration in seconds
        stride_sec:   hop between consecutive windows in seconds
        fps:          MERT frame rate

    Returns:
        win_embs:   [N, D] raw (un-normalized) mean-pooled window embeddings
        timestamps: [N]   center timestamp of each window in seconds
                          (= right edge, i.e. time of the most recent frame)
    """
    win_frames = max(1, int(round(lookback_sec * fps)))
    hop_frames = max(1, int(round(stride_sec * fps)))
    T = frame_embs.shape[0]

    rows: list[torch.Tensor] = []
    ts: list[float] = []

    # Start at first full window
    f = win_frames
    while f <= T:
        chunk = frame_embs[f - win_frames : f]  # [win_frames, D]
        rows.append(chunk.mean(dim=0))
        ts.append(f / fps)
        f += hop_frames

    if not rows:
        return torch.empty(0, frame_embs.shape[-1]), np.array([], dtype=np.float32)

    return torch.stack(rows, dim=0), np.array(ts, dtype=np.float32)


def pool_slide_embeddings(
    win_embs: torch.Tensor,     # [N, D] all-song window embeddings (raw)
    timestamps: np.ndarray,     # [N] timestamps corresponding to win_embs
    t_start: float,
    t_stop: float,
) -> torch.Tensor:
    """
    Pool the window embeddings whose timestamps fall in [t_start, t_stop].

    Returns a single [D] mean vector, or zeros if no windows fall in range.
    """
    mask = (timestamps >= t_start) & (timestamps < t_stop)
    if mask.sum() == 0:
        return torch.zeros(win_embs.shape[-1])
    return win_embs[torch.from_numpy(mask)].mean(dim=0)
