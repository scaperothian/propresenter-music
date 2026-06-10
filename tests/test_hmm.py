"""Tests for the online HMM predictor."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers: build a small 4-slide HMM
# ---------------------------------------------------------------------------

def _make_hmm(n=4, durations=None):
    from ppsync.hmm import HMMPredictor
    from ppsync.preprocess import build_hmm_transition

    if durations is None:
        durations = [5.0, 8.0, 6.0, 10.0]

    t_refs = np.cumsum([0.0] + durations[:-1])
    t_stops = t_refs + np.array(durations)
    A, pi = build_hmm_transition(t_refs, t_stops, stride_sec=0.2)
    return HMMPredictor(t_refs, t_stops, A, pi), t_refs, t_stops


# ---------------------------------------------------------------------------
# build_hmm_transition
# ---------------------------------------------------------------------------

def test_transition_rows_sum_to_one():
    from ppsync.preprocess import build_hmm_transition

    t_refs = np.array([0.0, 5.0, 12.0, 20.0])
    t_stops = np.array([5.0, 12.0, 20.0, 30.0])
    A, _ = build_hmm_transition(t_refs, t_stops, stride_sec=0.2)
    np.testing.assert_allclose(A.sum(axis=1), 1.0, atol=1e-9)


def test_transition_is_left_to_right():
    from ppsync.preprocess import build_hmm_transition

    t_refs = np.array([0.0, 5.0, 10.0])
    t_stops = np.array([5.0, 10.0, 20.0])
    A, _ = build_hmm_transition(t_refs, t_stops, stride_sec=0.2)
    # Only diagonal and super-diagonal should be non-zero
    N = len(t_refs)
    for i in range(N):
        for j in range(N):
            if j < i or j > i + 1:
                assert A[i, j] == pytest.approx(0.0, abs=1e-12), \
                    f"A[{i},{j}] = {A[i, j]} should be 0"


def test_last_state_absorbing():
    from ppsync.preprocess import build_hmm_transition

    t_refs = np.array([0.0, 5.0])
    t_stops = np.array([5.0, 15.0])
    A, _ = build_hmm_transition(t_refs, t_stops, stride_sec=0.2)
    assert A[-1, -1] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# HMMPredictor.update
# ---------------------------------------------------------------------------

def test_update_returns_required_keys():
    hmm, t_refs, _ = _make_hmm()
    out = hmm.update(obs_t=t_refs[1] + 1.0, dtw_confidence=0.9)
    for key in ("current_slide", "state_probs", "expected_pos_t",
                "predicted_next_t", "next_slide_idx", "trigger_confidence"):
        assert key in out


def test_state_probs_sum_to_one():
    hmm, t_refs, _ = _make_hmm()
    for t in t_refs:
        out = hmm.update(obs_t=t + 0.5, dtw_confidence=0.9)
        np.testing.assert_allclose(out["state_probs"].sum(), 1.0, atol=1e-9)


def test_confident_observation_moves_state():
    """Feeding observations clearly in slide 2 should push the HMM to state 2."""
    hmm, t_refs, t_stops = _make_hmm(n=4, durations=[5.0, 8.0, 6.0, 10.0])
    slide2_mid = (t_refs[2] + t_stops[2]) / 2
    for _ in range(20):  # repeated observations in slide 2
        out = hmm.update(obs_t=slide2_mid, dtw_confidence=1.0)
    assert out["current_slide"] == 2


def test_low_confidence_uses_transition_only():
    """With confidence=0, state should drift forward via the transition model."""
    hmm, t_refs, _ = _make_hmm()
    hmm.set_prior_from_coarse(slide_idx=0, confidence=1.0)
    for _ in range(100):
        out = hmm.update(obs_t=0.0, dtw_confidence=0.0)
    # After many steps with no observations, belief should spread rightward
    assert out["state_probs"][0] < 0.9  # not stuck at state 0


def test_reset_returns_uniform():
    hmm, t_refs, _ = _make_hmm()
    hmm.set_prior_from_coarse(0, confidence=1.0)
    hmm.reset(uniform=True)
    np.testing.assert_allclose(
        hmm.alpha, np.ones(4) / 4, atol=1e-12
    )


def test_set_prior_concentrates_mass():
    hmm, _, _ = _make_hmm()
    hmm.set_prior_from_coarse(slide_idx=2, confidence=0.9)
    assert hmm.alpha[2] > 0.85


def test_trigger_confidence_near_boundary():
    """Trigger confidence should be high when we're near the end of a slide."""
    hmm, t_refs, t_stops = _make_hmm(n=3, durations=[5.0, 5.0, 5.0])
    # Force belief onto slide 0, then observe very late in slide 0
    hmm.set_prior_from_coarse(0, confidence=1.0)
    near_end_t = t_stops[0] - 0.2
    for _ in range(5):
        out = hmm.update(obs_t=near_end_t, dtw_confidence=0.95)
    # trigger_confidence grows as we approach the boundary
    assert out["trigger_confidence"] > 0.3
