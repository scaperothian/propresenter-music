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
) -> dict:
    """
    Two-step alignment: cosine search → DTW refinement.

    Returns a dict with keys:
        candidate_t      float   coarse position from cosine search (seconds)
        refined_t        float   refined position from DTW (seconds)
        path_cost        float   DTW path cost
        confidence       float   0..1
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
    candidate_idx, sim = similarity_search(latest_live, ref_embs, lo_idx, hi_idx)
    candidate_t = float(ref_timestamps[candidate_idx])

    # --- Step 2: DTW refinement ---
    ctx_frames = max(len(live_buffer), int(dtw_context_sec / stride_sec))
    dtw_lo = max(0, candidate_idx - ctx_frames)
    dtw_hi = min(len(ref_embs), candidate_idx + ctx_frames)

    ref_window = ref_embs[dtw_lo:dtw_hi]    # [N_ctx, D]

    if len(ref_window) >= len(live_buffer) and len(live_buffer) > 0:
        end_rel, path_cost, confidence = subsequence_dtw(
            live_buffer, ref_window, band_ratio=band_ratio
        )
        refined_idx = dtw_lo + end_rel
        refined_t = float(ref_timestamps[min(refined_idx, len(ref_timestamps) - 1)])
    else:
        # Fallback: trust coarse search if window too small
        refined_t = candidate_t
        path_cost = float("inf")
        confidence = float(np.clip(sim, 0.0, 1.0))

    return {
        "candidate_t": candidate_t,
        "refined_t": refined_t,
        "path_cost": path_cost,
        "confidence": confidence,
        "search_lo_t": float(ref_timestamps[lo_idx]) if lo_idx < len(ref_timestamps) else search_lo_t,
        "search_hi_t": float(ref_timestamps[hi_idx - 1]) if hi_idx - 1 < len(ref_timestamps) else search_hi_t,
        "dtw_ref_lo_idx": dtw_lo,
        "dtw_ref_hi_idx": dtw_hi,
    }
