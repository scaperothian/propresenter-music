"""Online HMM predictor for slide state estimation.

Layer 3 in the pipeline.  Uses a left-to-right HMM where:
  - Hidden states  = slide indices (one per slide in the manifest)
  - Observations   = DTW position estimate in seconds
  - Transition matrix A  is built offline from JSON timestamps (see preprocess.py)
  - Emission model B(i, t) = Gaussian(t; mu_i, sigma_i)
      mu_i    = midpoint of slide i   (t_ref_i + t_stop_i) / 2
      sigma_i = max(slide_duration_i * OBS_SIGMA_RATIO, MIN_SIGMA_SEC)

The forward algorithm runs online: alpha[i] = P(current state = i | history).
When DTW confidence is below threshold, the observation is treated as missing
and the state distribution is propagated only through the transition model.

After each update, the predictor returns:
  current_slide     int     argmax state
  state_probs       [N]     posterior probabilities
  predicted_next_t  float   expected time of next slide boundary
  trigger_confidence float  probability that next transition is imminent
"""

from __future__ import annotations

import numpy as np

from .config import CONFIDENCE_THRESHOLD, HMM_MIN_SIGMA_SEC, HMM_OBS_SIGMA_RATIO


class HMMPredictor:
    """Online forward-filter HMM over slide states."""

    def __init__(
        self,
        slide_t_refs: np.ndarray,   # [N]
        slide_t_stops: np.ndarray,  # [N]
        hmm_A: np.ndarray,          # [N, N] transition matrix (row = from, col = to)
        hmm_pi: np.ndarray,         # [N] initial distribution
        obs_sigma_ratio: float = HMM_OBS_SIGMA_RATIO,
        min_sigma_sec: float = HMM_MIN_SIGMA_SEC,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self.N = len(slide_t_refs)
        self.slide_t_refs = slide_t_refs.astype(np.float64)
        self.slide_t_stops = slide_t_stops.astype(np.float64)
        self.A = hmm_A.astype(np.float64)     # [N, N]
        self.pi = hmm_pi.astype(np.float64)   # [N]
        self.confidence_threshold = confidence_threshold

        # Pre-compute emission parameters
        durations = slide_t_stops - slide_t_refs
        self.mu = (slide_t_refs + slide_t_stops) / 2.0          # [N]
        self.sigma = np.maximum(durations * obs_sigma_ratio, min_sigma_sec)  # [N]

        # Running forward variable (un-normalized)
        self.alpha = self.pi.copy()
        self._step = 0

    def reset(self, uniform: bool = True) -> None:
        """Reset belief to prior (uniform or original pi)."""
        self.alpha = np.ones(self.N, dtype=np.float64) / self.N if uniform else self.pi.copy()
        self._step = 0

    def set_prior_from_coarse(self, slide_idx: int, confidence: float = 1.0) -> None:
        """
        Seed the belief distribution from a coarse MERT slide classification.

        Places *confidence* probability mass on *slide_idx* and distributes
        the remainder uniformly.
        """
        uniform_weight = (1.0 - confidence) / self.N
        self.alpha = np.full(self.N, uniform_weight, dtype=np.float64)
        self.alpha[slide_idx] += confidence
        self.alpha /= self.alpha.sum()

    def _emission(self, obs_t: float) -> np.ndarray:
        """Gaussian emission probabilities for all states given obs_t."""
        diff = obs_t - self.mu
        log_b = -0.5 * (diff / self.sigma) ** 2
        b = np.exp(log_b - log_b.max())  # numerically stable
        return b / (b.sum() + 1e-300)

    def update(
        self,
        obs_t: float,
        dtw_confidence: float,
        coarse_slide_idx: int | None = None,
    ) -> dict:
        """
        One step of the forward filter.

        Args:
            obs_t:             DTW-refined position estimate in seconds
            dtw_confidence:    DTW confidence [0, 1]
            coarse_slide_idx:  MERT coarse slide classification (optional;
                               used to override observation when DTW is weak)

        Returns:
            dict with keys:
                current_slide       int
                state_probs         np.ndarray [N]
                expected_pos_t      float  (expected song position per HMM)
                predicted_next_t    float  (predicted next slide boundary)
                next_slide_id_idx   int
                trigger_confidence  float  (P(in penultimate or final frame of slide))
        """
        # --- Transition step: alpha_hat[j] = sum_i alpha[i] * A[i,j] ---
        alpha_pred = self.alpha @ self.A  # [N]

        # --- Emission update (only when DTW is confident) ---
        if dtw_confidence >= self.confidence_threshold:
            b = self._emission(obs_t)
            alpha_new = alpha_pred * b
        elif coarse_slide_idx is not None:
            # Weak DTW but strong coarse signal: use soft coarse prior
            b = np.zeros(self.N, dtype=np.float64)
            b[coarse_slide_idx] = 1.0
            alpha_new = alpha_pred * (0.3 * b + 0.7)  # partial update
        else:
            # No reliable observation: pure prediction
            alpha_new = alpha_pred

        # Normalize
        total = alpha_new.sum()
        if total > 1e-300:
            alpha_new /= total
        else:
            alpha_new = np.ones(self.N, dtype=np.float64) / self.N

        self.alpha = alpha_new
        self._step += 1

        current_slide = int(np.argmax(self.alpha))
        state_probs = self.alpha.copy()

        # --- Expected song position ---
        expected_pos_t = float(np.dot(self.alpha, self.mu))

        # --- Predict next slide boundary ---
        next_idx = min(current_slide + 1, self.N - 1)
        predicted_next_t = float(self.slide_t_refs[next_idx])

        # Trigger confidence: P(we are in current slide) * P(near boundary)
        p_current = float(self.alpha[current_slide])
        slide_dur = float(self.slide_t_stops[current_slide] - self.slide_t_refs[current_slide])
        time_left = max(0.0, predicted_next_t - obs_t)
        fraction_elapsed = 1.0 - (time_left / max(slide_dur, 0.001))
        trigger_confidence = float(p_current * max(0.0, fraction_elapsed))

        return {
            "current_slide": current_slide,
            "state_probs": state_probs,
            "expected_pos_t": expected_pos_t,
            "predicted_next_t": predicted_next_t,
            "next_slide_idx": next_idx,
            "trigger_confidence": trigger_confidence,
        }
