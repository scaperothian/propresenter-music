"""Tests for the subsequence DTW alignment module."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_l2(shape, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal(shape).astype(np.float32)
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    return X / (norms + 1e-9)


# ---------------------------------------------------------------------------
# cosine_distance_matrix
# ---------------------------------------------------------------------------

def test_cosine_distance_self_is_zero():
    from ppsync.dtw import cosine_distance_matrix

    X = _rand_l2((10, 64))
    C = cosine_distance_matrix(X, X)
    diag = np.diag(C)
    np.testing.assert_allclose(diag, 0.0, atol=1e-5)


def test_cosine_distance_range():
    from ppsync.dtw import cosine_distance_matrix

    A = _rand_l2((5, 32))
    B = _rand_l2((7, 32))
    C = cosine_distance_matrix(A, B)
    assert C.shape == (5, 7)
    assert (C >= 0).all()
    assert (C <= 2.01).all()


# ---------------------------------------------------------------------------
# subsequence_dtw
# ---------------------------------------------------------------------------

def test_subsequence_dtw_identical_subsequence():
    """When query equals a window of ref, best cost should be near zero."""
    from ppsync.dtw import subsequence_dtw

    M, D = 20, 64
    rng = np.random.default_rng(0)
    ref = _rand_l2((100, D))
    # Inject query at position 30–50
    query = ref[30:50].copy()

    end_idx, cost, conf = subsequence_dtw(query, ref)
    assert end_idx == pytest.approx(49, abs=3), f"expected ~49, got {end_idx}"
    assert conf > 0.8


def test_subsequence_dtw_confidence_range():
    from ppsync.dtw import subsequence_dtw

    query = _rand_l2((10, 32))
    ref = _rand_l2((50, 32))
    _, cost, conf = subsequence_dtw(query, ref)
    assert 0.0 <= conf <= 1.0


def test_subsequence_dtw_empty_inputs():
    from ppsync.dtw import subsequence_dtw

    empty = np.zeros((0, 32), dtype=np.float32)
    ref = _rand_l2((10, 32))
    _, cost, conf = subsequence_dtw(empty, ref)
    assert cost == float("inf")
    assert conf == 0.0


# ---------------------------------------------------------------------------
# Step penalty (slope constraint)
# ---------------------------------------------------------------------------

def test_step_penalty_zero_is_plain_dtw():
    """penalty=0 must be identical to the default (no-penalty) call."""
    from ppsync.dtw import subsequence_dtw

    ref = _rand_l2((80, 48))
    query = ref[20:44].copy()
    a = subsequence_dtw(query, ref)
    b = subsequence_dtw(query, ref, step_penalty=0.0)
    assert a == b


def test_step_penalty_leaves_clean_diagonal_match_unchanged():
    """A query that is an exact diagonal subsequence needs no warping, so the
    free diagonal path is optimal regardless of penalty — end index and cost
    must be invariant.  (This is why studio playback is unaffected.)"""
    from ppsync.dtw import subsequence_dtw

    ref = _rand_l2((100, 64))
    query = ref[30:50].copy()
    end0, cost0, _ = subsequence_dtw(query, ref, step_penalty=0.0)
    end_hi, cost_hi, _ = subsequence_dtw(query, ref, step_penalty=1.0)
    assert end_hi == end0
    assert cost_hi == pytest.approx(cost0, abs=1e-6)


def test_step_penalty_raises_cost_on_warped_match():
    """When matching requires stalling (a repeated/stretched ref frame), the
    penalty makes the warped path cost strictly more — the lever that pushes
    the chosen path toward the diagonal."""
    from ppsync.dtw import subsequence_dtw

    ref = _rand_l2((60, 48))
    # Query traverses ref 20..30 but stalls on frame 25 (repeated 4x) —
    # plain DTW absorbs this with horizontal moves at zero extra cost.
    idths = list(range(20, 26)) + [25, 25, 25] + list(range(26, 31))
    query = ref[idths].copy()
    _, cost0, _ = subsequence_dtw(query, ref, step_penalty=0.0)
    _, cost_pen, _ = subsequence_dtw(query, ref, step_penalty=0.2)
    assert cost_pen > cost0 + 0.1  # the stalls now cost real penalty


# ---------------------------------------------------------------------------
# similarity_search
# ---------------------------------------------------------------------------

def test_similarity_search_finds_best():
    from ppsync.dtw import similarity_search

    D = 64
    ref = _rand_l2((100, D))
    # query is exactly ref[42]
    query = ref[42].copy()
    idx, sim = similarity_search(query, ref, 0, 100)
    assert idx == 42
    assert sim > 0.99


def test_similarity_search_respects_bounds():
    from ppsync.dtw import similarity_search

    D = 32
    ref = _rand_l2((100, D))
    query = ref[10].copy()
    # Best match is at 10 but we restrict search to [50, 80]
    idx, _ = similarity_search(query, ref, 50, 80)
    assert 50 <= idx < 80


# ---------------------------------------------------------------------------
# align (full two-step)
# ---------------------------------------------------------------------------

def test_align_returns_expected_keys():
    from ppsync.dtw import align

    D = 64
    ref = _rand_l2((500, D))
    ts = np.arange(500, dtype=np.float32) * 0.02  # 20ms stride
    live = _rand_l2((20, D))

    result = align(
        live_buffer=live,
        ref_embs=ref,
        ref_timestamps=ts,
        search_lo_t=0.0,
        search_hi_t=float(ts[-1]),
        dtw_context_sec=2.0,
    )
    for key in ("candidate_t", "refined_t", "path_cost", "confidence"):
        assert key in result

    assert 0.0 <= result["confidence"] <= 1.0
    assert result["refined_t"] >= 0.0
