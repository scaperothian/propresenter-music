"""Main alignment pipeline: MERT → DTW → HMM → Trigger.

One SongAligner instance processes audio chunks in order and maintains all
layer state.  Call process_chunk() for each new audio chunk; it returns a
telemetry dict for every frame.

Live MERT embedding
-------------------
Incoming audio chunks fill a ring buffer holding the last LOOKBACK_SEC of
raw audio.  Each update re-embeds the whole window in one MERT call and
mean-pools the frames, giving the single embedding that feeds the cosine
search and DTW.  This matches the sliding-window reference preprocessing
and is independent of chunk phase.

DTW live buffer
---------------
The pooled embeddings from the last DTW_LIVE_SEC seconds are stacked into a
matrix and compared against the reference window around the coarse candidate.

Search window management
------------------------
Once alignment is initialised, the forward search window is anchored at the
last confirmed position (no backward search).  Before initialisation the
entire reference sequence is eligible.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from .config import (
    CHUNK_SEC,
    CONFIDENCE_THRESHOLD,
    DTW_LIVE_SEC,
    DTW_SEARCH_SEC,
    LOOKBACK_SEC,
    MERT_FRAME_RATE,
    MERT_LAYER,
    SILENCE_RMS_DBFS,
    TARGET_SR,
    TRIGGER_BUFFER_MS,
    TRIGGER_CONFIDENCE_MIN,
    TRIGGER_LATE_GRACE_SEC,
)
from .dtw import align as dtw_align
from .embed import embed_chunk_live
from .hmm import HMMPredictor
from .preprocess import load_cache
from .transform import apply_contrastive
from .trigger import TriggerScheduler


class SongAligner:
    """
    Full alignment pipeline for one song.

    Args:
        cache_path:       path to the .npz cache produced by preprocess_song()
        rest_url:         REST endpoint for slide triggers
        trigger_buffer_ms: fire trigger this many ms before slide boundary
        trigger_conf_min: minimum HMM confidence to fire trigger
        dry_run:          log triggers but don't send HTTP requests
        dtw_live_sec:     live embedding buffer duration fed to DTW
        dtw_search_sec:   forward search window in reference audio
        chunk_sec:        expected audio chunk size (must match AudioCapture)
    """

    def __init__(
        self,
        cache_path: Path,
        model=None,
        processor=None,
        device: str = "cpu",
        rest_url: str = "http://localhost:5000/slide",
        trigger_buffer_ms: float = TRIGGER_BUFFER_MS,
        trigger_conf_min: float = TRIGGER_CONFIDENCE_MIN,
        dry_run: bool = False,
        dtw_live_sec: float = DTW_LIVE_SEC,
        dtw_search_sec: float = DTW_SEARCH_SEC,
        chunk_sec: float = CHUNK_SEC,
    ) -> None:
        self.device = device
        self.model = model
        self.processor = processor
        self.chunk_sec = chunk_sec
        self.dtw_live_sec = dtw_live_sec
        self.dtw_search_sec = dtw_search_sec

        # Load cache
        cache = load_cache(cache_path)
        self.ref_embs: np.ndarray = cache["ref_embs"]          # [N_ref, D]
        self.ref_timestamps: np.ndarray = cache["ref_timestamps"]  # [N_ref]
        self.slide_protos: np.ndarray = cache["slide_protos"]  # [N_slides, D]
        self.slide_ids: list[str] = cache["slide_ids"]
        self.slide_t_refs: np.ndarray = cache["slide_t_refs"]
        self.slide_t_stops: np.ndarray = cache["slide_t_stops"]
        self.global_emb: np.ndarray = cache["global_emb"]      # [D]
        self.global_emb_t = torch.from_numpy(self.global_emb)
        self.song_duration: float = float(cache["song_duration"])
        self.stride_sec: float = float(cache["stride_sec"])
        self.mert_layer: int = int(cache["mert_layer"])
        # Effective MERT frame rate of the cache (short live chunks lose conv
        # edge frames, so this is below the nominal MERT_FRAME_RATE).  Live
        # pooling must span the same number of frames the reference used.
        self.frame_rate: float = float(cache.get("frame_rate", MERT_FRAME_RATE))

        # HMM
        self.hmm = HMMPredictor(
            slide_t_refs=cache["slide_t_refs"],
            slide_t_stops=cache["slide_t_stops"],
            hmm_A=cache["hmm_A"],
            hmm_pi=cache["hmm_pi"],
            confidence_threshold=CONFIDENCE_THRESHOLD,
        )

        # Trigger
        self.trigger = TriggerScheduler(
            rest_url=rest_url,
            buffer_ms=trigger_buffer_ms,
            confidence_min=trigger_conf_min,
            dry_run=dry_run,
        )

        # Audio ring buffer spanning the lookback window.  The whole window is
        # re-embedded in one MERT call per chunk so the live embedding is a
        # pure function of the last LOOKBACK_SEC of audio — independent of the
        # chunk phase.  (Caching per-chunk MERT frames breaks when playback
        # start is not aligned to the chunk grid: tracking drops 100% -> 4%
        # on a 0.1s phase shift.)
        self._audio_ring: np.ndarray = np.zeros(0, dtype=np.float32)
        self._lookback_samples = int(LOOKBACK_SEC * TARGET_SR)

        # Pooled embedding ring buffer for DTW
        max_dtw_embs = int(dtw_live_sec / chunk_sec) + 2
        self._dtw_emb_buffer: deque[np.ndarray] = deque(maxlen=max_dtw_embs)

        # Search window state
        self._search_lo_t: float = 0.0
        self._search_hi_t: float = float(self.ref_timestamps[-1]) if len(self.ref_timestamps) else dtw_search_sec
        self._initialized: bool = False
        self._confirmed_t: float = 0.0  # last high-confidence song position

        self._chunk_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Full reset for a new song or manual resync."""
        self._audio_ring = np.zeros(0, dtype=np.float32)
        self._dtw_emb_buffer.clear()
        self._search_lo_t = 0.0
        self._search_hi_t = float(self.ref_timestamps[-1]) if len(self.ref_timestamps) else self.dtw_search_sec
        self._initialized = False
        self._confirmed_t = 0.0
        self._chunk_count = 0
        self.hmm.reset()
        self.trigger.reset()

    def process_chunk(
        self,
        audio_chunk: np.ndarray,    # [N_samples] float32 @ TARGET_SR
        chunk_wall_t: float | None = None,  # wall-clock time of chunk start
    ) -> dict:
        """
        Process one audio chunk through the full pipeline.

        Returns a telemetry dict with per-layer metrics.
        """
        t0 = time.perf_counter()
        if chunk_wall_t is None:
            chunk_wall_t = time.monotonic()

        # ---- Layer 0: MERT embedding -----------------------------------------
        # Reference embeddings were pooled over a FULL lookback window, so live
        # embeddings are only comparable once the ring spans the whole lookback
        # — partial windows must not enter the DTW buffer or they poison the
        # first DTW_LIVE_SEC of matches after start.
        audio = np.asarray(audio_chunk, dtype=np.float32)
        self._audio_ring = np.concatenate([self._audio_ring, audio])[-self._lookback_samples:]
        if len(self._audio_ring) < self._lookback_samples:
            self._chunk_count += 1
            return {"status": "buffering", "chunk": self._chunk_count}

        # Silence gate: ambient noise still DTW-matches the song's quietest
        # section with above-threshold confidence, so an open mic before the
        # song starts could walk the position into a boundary.  Idle instead.
        rms_dbfs = 20.0 * np.log10(float(np.sqrt(np.mean(self._audio_ring ** 2))) + 1e-12)
        if rms_dbfs < SILENCE_RMS_DBFS:
            self._chunk_count += 1
            return {"status": "silence", "chunk": self._chunk_count,
                    "rms_dbfs": round(rms_dbfs, 1)}

        frames = self._embed_chunk(torch.from_numpy(self._audio_ring))  # [T, D]
        pooled_raw = frames.mean(dim=0)  # [D]
        pooled_norm = apply_contrastive(pooled_raw, self.global_emb_t).numpy()  # [D]
        self._dtw_emb_buffer.append(pooled_norm)

        # DTW on a near-empty buffer matches noise confidently (a 1-frame query
        # fits anywhere); hold off all alignment layers until the buffer is full.
        if len(self._dtw_emb_buffer) < self._dtw_emb_buffer.maxlen:
            self._chunk_count += 1
            return {"status": "buffering", "chunk": self._chunk_count}

        # ---- Layer 1: MERT coarse alignment (slide prototypes) ---------------
        sims = self.slide_protos @ pooled_norm  # [N_slides]
        coarse_slide_idx = int(np.argmax(sims))
        coarse_confidence = float(sims[coarse_slide_idx])

        # On first confident coarse match, seed the HMM and narrow the window
        if not self._initialized and coarse_confidence > CONFIDENCE_THRESHOLD:
            self._confirmed_t = float(self.slide_t_refs[coarse_slide_idx])
            self._search_lo_t = max(0.0, self._confirmed_t - 5.0)
            self._search_hi_t = self._confirmed_t + self.dtw_search_sec
            self.hmm.set_prior_from_coarse(coarse_slide_idx, confidence=coarse_confidence)
            self._initialized = True

        # ---- Layer 2: Subsequence DTW ----------------------------------------
        live_buffer = np.stack(list(self._dtw_emb_buffer))  # [M, D]
        dtw_ctx_sec = max(self.dtw_live_sec * 2, 10.0)

        dtw_result = dtw_align(
            live_buffer=live_buffer,
            ref_embs=self.ref_embs,
            ref_timestamps=self.ref_timestamps,
            search_lo_t=self._search_lo_t,
            search_hi_t=self._search_hi_t,
            dtw_context_sec=dtw_ctx_sec,
            band_ratio=0.1,
        )
        refined_t = dtw_result["refined_t"]
        dtw_confidence = dtw_result["confidence"]

        # Advance search window anchor when DTW is confident
        if dtw_confidence >= CONFIDENCE_THRESHOLD and refined_t > self._confirmed_t:
            self._confirmed_t = refined_t
            self._search_lo_t = max(0.0, refined_t - 2.0)
            self._search_hi_t = refined_t + self.dtw_search_sec

        # ---- Layer 3: HMM predictor ------------------------------------------
        hmm_out = self.hmm.update(
            obs_t=refined_t,
            dtw_confidence=dtw_confidence,
            coarse_slide_idx=coarse_slide_idx,
        )
        current_slide = hmm_out["current_slide"]
        trigger_conf = hmm_out["trigger_confidence"]
        predicted_next_t = hmm_out["predicted_next_t"]

        # ---- Trigger ---------------------------------------------------------
        # Drive the trigger from the DTW position when it is confident: the
        # HMM's expected_pos_t is a probability-weighted average of slide
        # MIDPOINTS, so it structurally lags the true position and crosses a
        # boundary only after the boundary has already passed.  The HMM path
        # remains the fallback for low-confidence stretches.
        if dtw_confidence >= CONFIDENCE_THRESHOLD:
            pos_t = refined_t
            pos_conf = dtw_confidence
        else:
            pos_t = hmm_out["expected_pos_t"]
            pos_conf = trigger_conf

        # Aim at the first boundary not yet fired.  Boundaries left more than
        # the grace window behind (mid-song join, long low-confidence gap) are
        # skipped; ones crossed just now fire late rather than never — DTW
        # jitter can step pos_t past a boundary between two chunks.  Only a
        # confident position estimate may consume boundaries.
        triggered = False
        triggered_slide_id = None
        boundary_idx = len(self.slide_ids)
        if pos_conf >= self.trigger.confidence_min:
            boundary_idx = self.trigger.last_triggered_idx + 1
            while (boundary_idx < len(self.slide_ids)
                   and pos_t > self.slide_t_refs[boundary_idx] + TRIGGER_LATE_GRACE_SEC):
                self.trigger.mark_skipped(boundary_idx)
                boundary_idx += 1
        if boundary_idx < len(self.slide_ids):
            triggered = self.trigger.update(
                current_song_t=pos_t,
                next_slide_idx=boundary_idx,
                next_slide_t=float(self.slide_t_refs[boundary_idx]),
                slide_id=self.slide_ids[boundary_idx],
                trigger_confidence=pos_conf,
                wall_time=chunk_wall_t,
            )
            if triggered:
                triggered_slide_id = self.slide_ids[boundary_idx]

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._chunk_count += 1

        return {
            "chunk": self._chunk_count,
            "wall_t": chunk_wall_t,
            # MERT
            "coarse_slide_idx": coarse_slide_idx,
            "coarse_slide_id": self.slide_ids[coarse_slide_idx],
            "coarse_confidence": round(coarse_confidence, 4),
            # DTW
            "dtw_candidate_t": round(dtw_result["candidate_t"], 3),
            "dtw_refined_t": round(refined_t, 3),
            "dtw_path_cost": round(dtw_result["path_cost"], 4) if dtw_result["path_cost"] != float("inf") else None,
            "dtw_confidence": round(dtw_confidence, 4),
            "dtw_search_lo": round(dtw_result["search_lo_t"], 3),
            "dtw_search_hi": round(dtw_result["search_hi_t"], 3),
            # HMM
            "hmm_current_slide": current_slide,
            "hmm_current_slide_id": self.slide_ids[current_slide],
            "hmm_expected_pos_t": round(hmm_out["expected_pos_t"], 3),
            "hmm_predicted_next_t": round(predicted_next_t, 3),
            "hmm_trigger_confidence": round(trigger_conf, 4),
            # Trigger
            "triggered": triggered,
            "triggered_slide_id": triggered_slide_id,
            # Perf
            "processing_ms": round(elapsed_ms, 1),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        """Run MERT on one audio chunk, return [T_chunk, D] for self.mert_layer."""
        if self.model is None or self.processor is None:
            raise RuntimeError("MERT model not loaded — pass model/processor to SongAligner.")
        hidden = embed_chunk_live(chunk, self.model, self.processor, self.device)
        return hidden[self.mert_layer]  # [T_chunk, D]
