"""Rigid (no-warp) alignment — the default matcher.

Slides the live query across the reference with the time mapping FIXED at 1:1.
Playback of a fixed recording does not warp time — only the acoustic channel
differs — so unlike DTW there is no path to bend, which keeps position
estimates unbiased and candidate margins sharp under mic/PA coloration.

Peer to ``dtw.py``: both expose a matcher with the same return-dict contract
(see ``rigid_align`` below and ``dtw.align``); ``aligner`` selects between
them on the ``MATCHER`` config.  All embeddings arriving here must already be
L2-normalized (contrastive transform applied) so cosine is a plain dot product.
"""

from __future__ import annotations

import numpy as np


def rigid_align(
    live_buffer: np.ndarray,     # [M, D] live embeddings, oldest→newest, L2-normalized
    ref_embs: np.ndarray,        # [N_ref, D] reference sequence (L2-normalized)
    ref_timestamps: np.ndarray,  # [N_ref] timestamps in seconds
    search_lo_t: float,
    search_hi_t: float,
    live_step: int,              # ref indices between consecutive live frames
    top_k: int = 1,
    cand_min_sep_sec: float = 8.0,
) -> dict:
    """
    Slide the live query across the reference with the time mapping FIXED at
    1:1 (playback of a fixed recording does not warp time — only the acoustic
    channel differs).  score(e) = mean_i cos(live[i], ref[e - (M-1-i)*step]).

    Unlike DTW, acoustic mismatch cannot be absorbed by bending the path, so
    position estimates stay unbiased and candidate margins stay sharp.

    Returns the same dict shape as dtw.align(); path_cost/cost_margin use
    per-frame (1 - cosine) units so the existing thresholds apply.
    """
    m = len(live_buffer)
    n = len(ref_embs)
    span = (m - 1) * live_step
    stride_sec = float(ref_timestamps[1] - ref_timestamps[0]) if n > 1 else 0.02

    lo_idx = max(int(np.searchsorted(ref_timestamps, search_lo_t, side="left")), span)
    hi_idx = min(int(np.searchsorted(ref_timestamps, search_hi_t, side="right")), n)
    if m == 0 or hi_idx <= lo_idx:
        return {"candidate_t": 0.0, "refined_t": 0.0, "path_cost": float("inf"),
                "confidence": 0.0, "cost_margin": 0.0,
                "search_lo_t": search_lo_t, "search_hi_t": search_hi_t,
                "dtw_ref_lo_idx": 0, "dtw_ref_hi_idx": 0}

    sims = ref_embs @ live_buffer.T                      # [N, M]
    ends = np.arange(lo_idx, hi_idx)                     # candidate end indices
    # row e -> indices [e - (M-1)*step, ..., e - step, e] for live 0..M-1
    offsets = (np.arange(m)[::-1] * live_step)           # [M], oldest first
    idx = ends[:, None] - offsets[None, :]               # [E, M]
    scores = sims[idx, np.arange(m)[None, :]].mean(axis=1)  # [E] mean cosine

    order = np.argsort(scores)[::-1]
    min_sep = max(1, int(cand_min_sep_sec / stride_sec))
    picks: list[int] = []
    for rel in order:
        if all(abs(int(rel) - p) >= min_sep for p in picks):
            picks.append(int(rel))
            if len(picks) >= max(1, top_k):
                break

    best = picks[0]
    best_score = float(scores[best])
    runner_score = float(scores[picks[1]]) if len(picks) > 1 else None
    refined_t = float(ref_timestamps[lo_idx + best])
    # Same units as DTW: per-frame cost = 1 - cosine; conf = (1 + cos) / 2.
    cost_margin = (best_score - runner_score) if runner_score is not None else 0.0

    return {
        "candidate_t": refined_t,
        "refined_t": refined_t,
        "path_cost": (1.0 - best_score) * m,
        "confidence": float(np.clip((1.0 + best_score) / 2.0, 0.0, 1.0)),
        "cost_margin": float(max(cost_margin, 0.0)),
        "search_lo_t": float(ref_timestamps[lo_idx]),
        "search_hi_t": float(ref_timestamps[hi_idx - 1]),
        "dtw_ref_lo_idx": lo_idx - span,
        "dtw_ref_hi_idx": hi_idx,
    }
