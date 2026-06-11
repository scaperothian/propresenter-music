"""Subsequence DTW alignment.

Layer 2 in the pipeline: given a buffer of live embeddings and a window of
reference embeddings (centred on the MERT coarse candidate), find the
best-matching subsequence alignment and return a refined position estimate.

All embeddings arriving here must already be L2-normalized (contrastive
transform already applied) so that the pairwise cost is simply 1 - dot.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Cost matrix
# ---------------------------------------------------------------------------

def cosine_distance_matrix(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Pairwise cosine distance matrix.

    Args:
        query: [M, D] L2-normalized row vectors (live buffer)
        ref:   [N, D] L2-normalized row vectors (reference window)

    Returns:
        [M, N] float32 cost matrix where cost = 1 - cosine_similarity
    """
    sim = query @ ref.T        # [M, N]
    return np.clip(1.0 - sim, 0.0, 2.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Subsequence DTW  (unconstrained start, constrained Sakoe-Chiba band)
# ---------------------------------------------------------------------------

def subsequence_dtw(
    query: np.ndarray,   # [M, D] live embeddings (L2-normalized)
    ref: np.ndarray,     # [N, D] reference window (L2-normalized, N > M)
    band_ratio: float = 0.1,
) -> tuple[int, float, float]:
    """
    Find the best-matching subsequence of *ref* for *query*.

    Uses the standard subsequence DTW formulation: the first query frame can
    align to ANY reference frame (free start), and we pick the end index that
    minimises the accumulated path cost.

    The *band_ratio* parameter is unused at the DP level because a Sakoe-Chiba
    band relative to the (0,0) diagonal would incorrectly block the optimal
    path (which is offset by the match position).  The caller in ``align()``
    already narrows the reference window around the coarse candidate, which
    provides equivalent computational savings.

    Args:
        query:      [M, D] live embedding buffer
        ref:        [N, D] reference window (should be longer than query)
        band_ratio: reserved for future use; currently ignored

    Returns:
        best_end_idx:  index in ref where the best subsequence ends
        path_cost:     total accumulated cost (lower = better)
        confidence:    1 - (path_cost / M / 2)  clipped to [0, 1]
    """
    M, N = len(query), len(ref)
    if M == 0 or N == 0:
        return 0, float("inf"), 0.0

    C = cosine_distance_matrix(query, ref)  # [M, N]

    # DP accumulation matrix
    D = np.full((M, N), np.inf, dtype=np.float64)

    # Subsequence DTW: first row initialises freely (any ref start)
    D[0, :] = C[0, :]

    for i in range(1, M):
        # Standard three-predecessor DP over all ref positions
        prev_row = D[i - 1, :]           # D[i-1, j]
        prev_diag = np.roll(prev_row, 1)  # D[i-1, j-1]
        prev_diag[0] = np.inf

        # D[i, j] = C[i,j] + min(D[i-1,j], D[i-1,j-1], D[i,j-1])
        # Compute row left-to-right to incorporate D[i, j-1] (horizontal move)
        row = C[i, :] + np.minimum(prev_row, prev_diag)
        for j in range(1, N):
            if row[j - 1] + C[i, j] < row[j]:
                row[j] = row[j - 1] + C[i, j]
        D[i, :] = row

    # Best end: argmin of last query row
    best_end_idx = int(np.argmin(D[M - 1, :]))
    path_cost = float(D[M - 1, best_end_idx])

    # Normalise: divide by M (query length) and by 2.0 (max cost per step)
    norm_cost = path_cost / max(M, 1) / 2.0
    confidence = float(np.clip(1.0 - norm_cost, 0.0, 1.0))

    return best_end_idx, path_cost, confidence


# ---------------------------------------------------------------------------
# Coarse similarity search  (step 1 — picks candidate before DTW)
# ---------------------------------------------------------------------------

def similarity_search(
    live_emb: np.ndarray,    # [D] single L2-normalized live embedding
    ref_embs: np.ndarray,    # [N_ref, D] full reference sequence
    search_lo: int,          # inclusive lower bound in ref index space
    search_hi: int,          # exclusive upper bound
) -> tuple[int, float]:
    """
    Find the reference index with highest cosine similarity to live_emb.

    Args:
        live_emb:  [D] L2-normalized
        ref_embs:  [N_ref, D] L2-normalized
        search_lo: restrict search to ref_embs[search_lo:search_hi]
        search_hi:

    Returns:
        best_idx:   absolute index into ref_embs
        similarity: cosine similarity at best_idx (in [-1, 1])
    """
    lo = max(0, search_lo)
    hi = min(len(ref_embs), search_hi)
    if lo >= hi:
        lo, hi = 0, len(ref_embs)

    window = ref_embs[lo:hi]  # [W, D]
    sims = window @ live_emb  # [W]
    rel_best = int(np.argmax(sims))
    return lo + rel_best, float(sims[rel_best])


def topk_candidates(
    live_emb: np.ndarray,    # [D] single L2-normalized live embedding
    ref_embs: np.ndarray,    # [N_ref, D]
    search_lo: int,
    search_hi: int,
    k: int,
    min_sep: int,            # minimum index separation between candidates
) -> list[int]:
    """
    Top-k cosine peaks with non-max suppression.

    Repeated sections (chorus, riff-based verses/outros) produce several
    near-equal cosine peaks; returning all of them lets DTW disambiguate
    using the full live buffer instead of trusting the single best 2s match.
    """
    lo = max(0, search_lo)
    hi = min(len(ref_embs), search_hi)
    if lo >= hi:
        lo, hi = 0, len(ref_embs)

    sims = ref_embs[lo:hi] @ live_emb  # [W]
    order = np.argsort(sims)[::-1]
    picks: list[int] = []
    for rel in order:
        if all(abs(int(rel) - (p - lo)) >= min_sep for p in picks):
            picks.append(lo + int(rel))
            if len(picks) == k:
                break
    return picks


# ---------------------------------------------------------------------------
# Rigid (no-warp) alignment — for linear playback of a fixed recording
# ---------------------------------------------------------------------------

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

    Returns the same dict shape as align(); path_cost/cost_margin use
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


# ---------------------------------------------------------------------------
# Full DTW alignment step (step 1 + step 2)
# ---------------------------------------------------------------------------

def align(
    live_buffer: np.ndarray,  # [M, D] recent live embeddings (L2-normalized)
    ref_embs: np.ndarray,     # [N_ref, D] full reference sequence (L2-normalized)
    ref_timestamps: np.ndarray,  # [N_ref] timestamps in seconds
    search_lo_t: float,       # lower bound of search window (seconds)
    search_hi_t: float,       # upper bound of search window (seconds)
    dtw_context_sec: float,   # half-width of DTW reference window around candidate
    band_ratio: float = 0.1,
    top_k: int = 1,
    cand_min_sep_sec: float = 8.0,
) -> dict:
    """
    Two-step alignment: cosine search → DTW refinement.

    With ``top_k > 1`` the cosine search returns several well-separated
    candidate regions and each is DTW-refined; the lowest path cost wins.
    Use this when the search window is wide (initial lock) — for repetitive
    songs the single best 2s cosine match often lands on the wrong repeat,
    and only the multi-second DTW query can tell them apart.

    Returns a dict with keys:
        candidate_t      float   coarse position from cosine search (seconds)
        refined_t        float   refined position from DTW (seconds)
        path_cost        float   DTW path cost
        confidence       float   0..1
        cost_margin      float   runner-up minus best normalized path cost
                                 (0 when only one candidate was evaluated)
        search_lo_t      float   bounds used
        search_hi_t      float
        dtw_ref_lo_idx   int     reference slice used for DTW
        dtw_ref_hi_idx   int
    """
    stride_sec = float(ref_timestamps[1] - ref_timestamps[0]) if len(ref_timestamps) > 1 else 0.02

    # Map time bounds to index bounds.  ref_timestamps does not start at 0
    # (the first window ends at lookback_sec), so search by value rather than
    # dividing by the stride.
    lo_idx = int(np.searchsorted(ref_timestamps, search_lo_t, side="left"))
    hi_idx = int(np.searchsorted(ref_timestamps, search_hi_t, side="right"))

    # --- Step 1: coarse cosine search ---
    latest_live = live_buffer[-1]  # most recent embedding
    if top_k > 1:
        min_sep = max(1, int(cand_min_sep_sec / stride_sec))
        candidates = topk_candidates(latest_live, ref_embs, lo_idx, hi_idx,
                                     k=top_k, min_sep=min_sep)
    else:
        candidate_idx, _sim = similarity_search(latest_live, ref_embs, lo_idx, hi_idx)
        candidates = [candidate_idx]

    # --- Step 2: DTW refinement per candidate, lowest path cost wins ---
    ctx_frames = max(len(live_buffer), int(dtw_context_sec / stride_sec))
    results = []  # (norm_cost, refined_t, path_cost, confidence, dtw_lo, dtw_hi, cand_t)
    for cand_idx in candidates:
        dtw_lo = max(0, cand_idx - ctx_frames)
        dtw_hi = min(len(ref_embs), cand_idx + ctx_frames)
        ref_window = ref_embs[dtw_lo:dtw_hi]    # [N_ctx, D]

        if len(ref_window) >= len(live_buffer) and len(live_buffer) > 0:
            end_rel, path_cost, confidence = subsequence_dtw(
                live_buffer, ref_window, band_ratio=band_ratio
            )
            refined_idx = dtw_lo + end_rel
            refined_t = float(ref_timestamps[min(refined_idx, len(ref_timestamps) - 1)])
            norm_cost = path_cost / max(len(live_buffer), 1)
        else:
            # Fallback: trust coarse search if window too small
            refined_t = float(ref_timestamps[cand_idx])
            path_cost = float("inf")
            sims_here = float(ref_embs[cand_idx] @ latest_live)
            confidence = float(np.clip(sims_here, 0.0, 1.0))
            norm_cost = float("inf")
        results.append((norm_cost, refined_t, path_cost, confidence,
                        dtw_lo, dtw_hi, float(ref_timestamps[cand_idx])))

    results.sort(key=lambda r: r[0])
    norm_cost, refined_t, path_cost, confidence, dtw_lo, dtw_hi, candidate_t = results[0]
    cost_margin = (results[1][0] - norm_cost) if len(results) > 1 else 0.0
    if not np.isfinite(cost_margin):
        cost_margin = 0.0

    return {
        "candidate_t": candidate_t,
        "refined_t": refined_t,
        "path_cost": path_cost,
        "confidence": confidence,
        "cost_margin": float(cost_margin),
        "search_lo_t": float(ref_timestamps[lo_idx]) if lo_idx < len(ref_timestamps) else search_lo_t,
        "search_hi_t": float(ref_timestamps[hi_idx - 1]) if hi_idx - 1 < len(ref_timestamps) else search_hi_t,
        "dtw_ref_lo_idx": dtw_lo,
        "dtw_ref_hi_idx": dtw_hi,
    }
