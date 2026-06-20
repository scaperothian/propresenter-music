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
    DTW_MIN_LIVE_SEC,
    DTW_SEARCH_SEC,
    INIT_AGREE_SEC,
    INIT_BUFFER_SEC,
    INIT_CAND_SEP_SEC,
    INIT_CONSISTENT_FRAMES,
    INIT_MIN_COST_MARGIN,
    INIT_TOP_K,
    JUMP_AGREE_SEC,
    JUMP_CONFIRM_FRAMES,
    JUMP_GUARD_SEC,
    JUMP_MIN_COST_MARGIN,
    LIVE_MEAN_ADAPT_SEC,
    LOOKBACK_SEC,
    MATCHER,
    MERT_FP16,
    MERT_FRAME_RATE,
    MERT_LAYER,
    SILENCE_RMS_DBFS,
    TARGET_SR,
    TRIGGER_BUFFER_MS,
    TRIGGER_CONFIDENCE_MIN,
)
from .dtw import align as dtw_align
from .dtw import rigid_align
from .embed import embed_chunk_live
from .hmm import HMMPredictor
from .preprocess import load_cache
from .trigger import TriggerScheduler


def select_trigger_boundary(
    last_triggered_idx: int,
    slide_t_refs: np.ndarray,
    pos_t: float,
) -> tuple[list[int], int]:
    """
    Pick which slide boundary the trigger should aim at.

    The target is the slide instance containing *pos_t* (catch-up: on a
    mid-song join or after a low-confidence gap the CURRENT slide must be
    shown immediately, not silently skipped while waiting for the next
    boundary).  Instances strictly before it are returned as skips.  If
    everything up to pos_t has already fired, aim at the next boundary.

    Returns:
        (skip_indices, boundary_idx)
    """
    current = int(np.searchsorted(slide_t_refs, pos_t, side="right")) - 1
    first_unfired = last_triggered_idx + 1
    skips = list(range(first_unfired, current))
    return skips, max(first_unfired, current)


def same_onscreen_slide(
    slide_t_refs: np.ndarray,
    slide_pp_indices: np.ndarray,
    t_a: float,
    t_b: float,
) -> bool:
    """
    True if reference times *t_a* and *t_b* land on slide instances that show
    the SAME ProPresenter slide — i.e. the same ``pp_slide_index``, identical
    on-screen text (a repeated chorus/refrain).  Used by the jump guard to
    refuse a jump that would not change the display: there is no benefit to
    moving the anchor to another instance of the slide already shown, only the
    risk of landing on the wrong repeat and skipping the sections between them.
    """
    def pp_at(t: float) -> int:
        i = max(0, int(np.searchsorted(slide_t_refs, t, side="right")) - 1)
        return int(slide_pp_indices[i])

    return pp_at(t_a) == pp_at(t_b)


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
        pp_controller=None,
        trigger_buffer_ms: float = TRIGGER_BUFFER_MS,
        trigger_conf_min: float = TRIGGER_CONFIDENCE_MIN,
        dry_run: bool = False,
        dtw_live_sec: float = DTW_LIVE_SEC,
        dtw_search_sec: float = DTW_SEARCH_SEC,
        chunk_sec: float = CHUNK_SEC,
        wall_timers: bool = True,
        matcher: str = MATCHER,
    ) -> None:
        self.device = device
        self.model = model
        self.processor = processor
        self.chunk_sec = chunk_sec
        self.dtw_live_sec = dtw_live_sec
        self.dtw_search_sec = dtw_search_sec
        self.matcher = matcher

        # Load cache
        cache = load_cache(cache_path)
        self.ref_embs: np.ndarray = cache["ref_embs"]          # [N_ref, D]
        self.ref_timestamps: np.ndarray = cache["ref_timestamps"]  # [N_ref]
        self.slide_protos: np.ndarray = cache["slide_protos"]  # [N_slides, D]
        self.slide_ids: list[str] = cache["slide_ids"]
        self.slide_t_refs: np.ndarray = cache["slide_t_refs"]
        self.slide_t_stops: np.ndarray = cache["slide_t_stops"]
        # ProPresenter slide position per manifest instance (repeats share
        # one); older caches lack it — fall back to chronological order.
        if "slide_pp_indices" in cache:
            self.slide_pp_indices: np.ndarray = cache["slide_pp_indices"]
        else:
            self.slide_pp_indices = np.arange(len(self.slide_ids), dtype=np.int32)
        self.pp_uuid: str = str(cache["pp_uuid"]) if "pp_uuid" in cache else ""
        # Song identity — older caches lack these; fall back to the cache
        # filename so logs always carry SOME song identifier.
        self.song_id: str = str(cache.get("song_id", "")) or Path(cache_path).stem
        self.artist: str = str(cache.get("artist", ""))
        self.song_slug: str = str(cache.get("song_slug", "")) or Path(cache_path).stem
        self.global_emb: np.ndarray = cache["global_emb"]      # [D]
        self.song_duration: float = float(cache["song_duration"])
        self.stride_sec: float = float(cache["stride_sec"])
        self.mert_layer: int = int(cache["mert_layer"])
        # Effective MERT frame rate of the cache (short live chunks lose conv
        # edge frames, so this is below the nominal MERT_FRAME_RATE).  Live
        # pooling must span the same number of frames the reference used.
        self.frame_rate: float = float(cache.get("frame_rate", MERT_FRAME_RATE))
        # Precision mismatch between cache and live MERT shifts the embedding
        # distribution — same failure mode as the chunk-context mismatch.
        cache_fp16 = bool(cache.get("mert_fp16", False))
        live_fp16 = MERT_FP16 and device != "cpu"
        if cache_fp16 != live_fp16:
            print(f"WARNING: cache built with fp16={cache_fp16} but live MERT runs "
                  f"fp16={live_fp16} — embeddings will not match well; "
                  f"re-run ppsync-preprocess.")

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
            pp_controller=pp_controller,
            wall_timers=wall_timers,
            # Crossings predicted within ~2.5 chunks get an exact-moment fire.
            schedule_horizon_sec=chunk_sec * 2.5,
        )

        # Audio ring buffer spanning the lookback window.  The whole window is
        # re-embedded in one MERT call per chunk so the live embedding is a
        # pure function of the last LOOKBACK_SEC of audio — independent of the
        # chunk phase.  (Caching per-chunk MERT frames breaks when playback
        # start is not aligned to the chunk grid: tracking drops 100% -> 4%
        # on a 0.1s phase shift.)
        self._audio_ring: np.ndarray = np.zeros(0, dtype=np.float32)
        self._lookback_samples = int(LOOKBACK_SEC * TARGET_SR)

        # Pooled embedding ring buffer for DTW.  DTW starts once the buffer
        # holds DTW_MIN_LIVE_SEC (a short query matches noise confidently).
        # PRE-LOCK the buffer grows to INIT_BUFFER_SEC — identical repeats
        # (chorus pairs) can only be told apart once the query spans a section
        # transition, so the initial decision needs more context than one
        # slide.  After lock it is trimmed to dtw_live_sec for cheap tracking.
        self._steady_dtw_embs = int(dtw_live_sec / chunk_sec)
        init_embs = max(int(INIT_BUFFER_SEC / chunk_sec), self._steady_dtw_embs)
        self._dtw_emb_buffer: deque[np.ndarray] = deque(maxlen=init_embs)
        self._min_dtw_embs = int(DTW_MIN_LIVE_SEC / chunk_sec)

        # Search window state
        self._search_lo_t: float = 0.0
        self._search_hi_t: float = float(self.ref_timestamps[-1]) if len(self.ref_timestamps) else dtw_search_sec
        self._initialized: bool = False
        self._confirmed_t: float = 0.0  # last high-confidence song position

        # Live-mean adaptation: running mean of raw live pooled embeddings,
        # blended over the cache's song mean as salient audio accumulates.
        # Mic/PA/room coloration shifts all live embeddings by a common
        # offset; centering live frames on their OWN mean cancels it.
        self._live_mean_sum: np.ndarray = np.zeros_like(self.global_emb, dtype=np.float64)
        self._live_mean_n: int = 0
        self._live_mean_full_n = max(1, int(LIVE_MEAN_ADAPT_SEC / chunk_sec))

        # Initial-lock consistency: repeated sections produce ambiguous
        # matches, so the first anchor needs several agreeing confident frames.
        self._init_hist: deque[float] = deque(maxlen=INIT_CONSISTENT_FRAMES)
        # Jump guard: a single confident frame must not move the anchor by
        # more than JUMP_GUARD_SEC — wrong-repeat matches look exactly like
        # this and the forward-only ratchet would lock them in.
        self._jump_hist: deque[float] = deque(maxlen=JUMP_CONFIRM_FRAMES)

        self._chunk_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Full reset for a new song or manual resync."""
        self._audio_ring = np.zeros(0, dtype=np.float32)
        init_embs = max(int(INIT_BUFFER_SEC / self.chunk_sec), self._steady_dtw_embs)
        self._dtw_emb_buffer = deque(maxlen=init_embs)
        self._search_lo_t = 0.0
        self._search_hi_t = float(self.ref_timestamps[-1]) if len(self.ref_timestamps) else self.dtw_search_sec
        self._initialized = False
        self._confirmed_t = 0.0
        self._init_hist.clear()
        self._jump_hist.clear()
        self._live_mean_sum = np.zeros_like(self.global_emb, dtype=np.float64)
        self._live_mean_n = 0
        self._chunk_count = 0
        self.hmm.reset()
        self.trigger.reset()

    def _advance_anchor(self, refined_t: float) -> None:
        """Move the confirmed position and re-center the search window."""
        self._confirmed_t = refined_t
        self._search_lo_t = max(0.0, refined_t - 2.0)
        self._search_hi_t = refined_t + self.dtw_search_sec

    def _match(self, live_buffer: np.ndarray, search_lo_t: float,
               search_hi_t: float, top_k: int) -> dict:
        """Run the configured matcher over [search_lo_t, search_hi_t]."""
        if self.matcher == "rigid":
            return rigid_align(
                live_buffer=live_buffer,
                ref_embs=self.ref_embs,
                ref_timestamps=self.ref_timestamps,
                search_lo_t=search_lo_t,
                search_hi_t=search_hi_t,
                live_step=max(1, round(self.chunk_sec / self.stride_sec)),
                top_k=top_k,
                cand_min_sep_sec=INIT_CAND_SEP_SEC,
            )
        return dtw_align(
            live_buffer=live_buffer,
            ref_embs=self.ref_embs,
            ref_timestamps=self.ref_timestamps,
            search_lo_t=search_lo_t,
            search_hi_t=search_hi_t,
            dtw_context_sec=max(self.dtw_live_sec * 2, 10.0),
            band_ratio=0.1,
            top_k=top_k,
            cand_min_sep_sec=INIT_CAND_SEP_SEC,
        )

    def _jump_beats_local(self, live_buffer: np.ndarray, jump_result: dict) -> bool:
        """
        True when the pending jump target's match cost beats a re-alignment
        restricted to the neighbourhood of the current anchor by
        JUMP_MIN_COST_MARGIN (normalized per query frame).

        jump_result is the window-wide best alignment (= the jump target).
        If the local hypothesis explains the live audio almost as well, the
        "jump" is a wrong-repeat match and must be rejected.
        """
        jump_cost = jump_result["path_cost"]
        if not np.isfinite(jump_cost):
            return False
        local = self._match(
            live_buffer,
            max(0.0, self._confirmed_t - 2.0),
            self._confirmed_t + JUMP_GUARD_SEC,
            top_k=1,
        )
        if not np.isfinite(local["path_cost"]):
            return True  # no usable local hypothesis (e.g. song end)
        m = max(len(live_buffer), 1)
        return (jump_cost / m) + JUMP_MIN_COST_MARGIN <= (local["path_cost"] / m)

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
        pooled_raw = frames.mean(dim=0).numpy()  # [D] raw (un-normalized)
        self._dtw_emb_buffer.append(pooled_raw)

        # Blend the normalization mean from the cache's song mean toward the
        # live stream's own running mean.  Mic/PA/room coloration offsets all
        # live embeddings alike; centering on the live mean cancels it (the
        # reference side is likewise centered on its own song mean).  The
        # whole buffer is re-normalized with the current mean each frame so
        # the DTW query stays internally consistent.
        self._live_mean_sum += pooled_raw.astype(np.float64)
        self._live_mean_n += 1
        w = min(1.0, self._live_mean_n / self._live_mean_full_n)
        live_mean = ((1.0 - w) * self.global_emb.astype(np.float64)
                     + w * (self._live_mean_sum / self._live_mean_n))

        def _contrast(arr: np.ndarray) -> np.ndarray:
            centered = arr - live_mean
            norms = np.linalg.norm(centered, axis=-1, keepdims=True)
            return (centered / np.maximum(norms, 1e-9)).astype(np.float32)

        pooled_norm = _contrast(pooled_raw)

        # DTW on a near-empty buffer matches noise confidently (a 1-frame query
        # fits anywhere); hold off all alignment until it has minimum context.
        if len(self._dtw_emb_buffer) < self._min_dtw_embs:
            self._chunk_count += 1
            return {"status": "buffering", "chunk": self._chunk_count}

        # ---- Layer 1: MERT coarse alignment (slide prototypes) ---------------
        sims = self.slide_protos @ pooled_norm  # [N_slides]
        coarse_slide_idx = int(np.argmax(sims))
        coarse_confidence = float(sims[coarse_slide_idx])

        # ---- Layer 2: sequence matching (DTW or rigid) ------------------------
        live_buffer = _contrast(np.stack(list(self._dtw_emb_buffer)))  # [M, D]

        # Pre-lock the search is song-wide and repeats are ambiguous:
        # refine several separated candidates, best cost wins.
        dtw_result = self._match(
            live_buffer, self._search_lo_t, self._search_hi_t,
            top_k=1 if self._initialized else INIT_TOP_K,
        )
        refined_t = dtw_result["refined_t"]
        dtw_confidence = dtw_result["confidence"]

        # ---- Anchor management: initial lock + jump guard ---------------------
        # obs_accepted: this frame's refined_t may be trusted downstream
        # (HMM observation, trigger position).  A confident frame is NOT
        # accepted while the initial lock or a large jump is unconfirmed.
        obs_accepted = False
        if dtw_confidence >= CONFIDENCE_THRESHOLD:
            if not self._initialized:
                # Initial lock: require consecutive confident frames agreeing
                # on position AND a clear best-vs-runner-up cost margin —
                # identical repeats (chorus pairs) tie on cost, and locking on
                # a tie picks an arbitrary instance.  Keep listening instead;
                # the growing query eventually spans a section transition.
                self._init_hist.append(refined_t)
                if (len(self._init_hist) == self._init_hist.maxlen
                        and max(self._init_hist) - min(self._init_hist) <= INIT_AGREE_SEC
                        and dtw_result["cost_margin"] >= INIT_MIN_COST_MARGIN):
                    self._initialized = True
                    obs_accepted = True
                    self._advance_anchor(refined_t)
                    slide_idx = max(0, int(np.searchsorted(
                        self.slide_t_refs, refined_t, side="right")) - 1)
                    self.hmm.set_prior_from_coarse(slide_idx, confidence=0.8)
                    # Trim the query buffer to the cheap steady-state size.
                    self._dtw_emb_buffer = deque(
                        list(self._dtw_emb_buffer)[-self._steady_dtw_embs:],
                        maxlen=self._steady_dtw_embs,
                    )
            elif refined_t - self._confirmed_t > JUMP_GUARD_SEC:
                # Same on-screen slide?  A jump whose target shows the SAME
                # ProPresenter slide as the current anchor (a repeated chorus/
                # refrain — identical text on the same pp_slide_index) changes
                # nothing on screen, so there is no benefit to taking it, only
                # the risk of landing on the wrong repeat and skipping the real
                # sections between them (observed: 11_chorus -> 14_chorus jumps
                # the bridge).  Hold position; the next DISTINGUISHING section
                # (a different pp_slide_index) is what legitimately re-acquires.
                if same_onscreen_slide(self.slide_t_refs, self.slide_pp_indices,
                                       refined_t, self._confirmed_t):
                    self._jump_hist.clear()
                else:
                    # Big forward jump to a different slide: wrong-repeat
                    # matches look exactly like this, and a stable wrong match
                    # agrees with itself — so on top of consecutive agreement,
                    # the jump target must BEAT a local re-alignment near the
                    # current anchor by a cost margin.
                    self._jump_hist.append(refined_t)
                    if (len(self._jump_hist) == self._jump_hist.maxlen
                            and max(self._jump_hist) - min(self._jump_hist) <= JUMP_AGREE_SEC
                            and self._jump_beats_local(live_buffer, dtw_result)):
                        obs_accepted = True
                        self._advance_anchor(refined_t)
                        self._jump_hist.clear()
            else:
                obs_accepted = True
                self._jump_hist.clear()
                if refined_t > self._confirmed_t:
                    self._advance_anchor(refined_t)
        else:
            self._init_hist.clear()
            self._jump_hist.clear()

        # ---- Layer 3: HMM predictor ------------------------------------------
        hmm_out = self.hmm.update(
            obs_t=refined_t,
            dtw_confidence=dtw_confidence if obs_accepted else 0.0,
            coarse_slide_idx=coarse_slide_idx,
        )
        current_slide = hmm_out["current_slide"]
        trigger_conf = hmm_out["trigger_confidence"]
        predicted_next_t = hmm_out["predicted_next_t"]

        # ---- Trigger ---------------------------------------------------------
        # Drive the trigger from the DTW position when it is confident and
        # accepted by the anchor logic: the HMM's expected_pos_t is a
        # probability-weighted average of slide MIDPOINTS, so it structurally
        # lags the true position.  The HMM path remains the fallback for
        # low-confidence stretches; nothing fires before the initial lock.
        if obs_accepted:
            pos_t = refined_t
            pos_conf = dtw_confidence
        else:
            pos_t = hmm_out["expected_pos_t"]
            pos_conf = trigger_conf

        # Aim at the slide instance containing pos_t (catch-up: a mid-song
        # join or a boundary stepped over by DTW jitter must still show the
        # CURRENT slide, immediately and at most once); instances strictly
        # before it are skipped.  Only a confident position estimate may
        # consume boundaries.
        triggered = False
        triggered_slide_id = None
        trigger_fire_t = None
        # Surface fires performed by scheduled timers since the last chunk.
        for ev in self.trigger.drain_fired():
            triggered = True
            triggered_slide_id = ev["slide_id"]
            trigger_fire_t = ev["fire_at_song_t"]
        boundary_idx = len(self.slide_ids)
        if self._initialized and pos_conf >= self.trigger.confidence_min:
            skips, boundary_idx = select_trigger_boundary(
                self.trigger.last_triggered_idx, self.slide_t_refs, pos_t
            )
            for k in skips:
                self.trigger.mark_skipped(k)
        if boundary_idx < len(self.slide_ids):
            fired_now = self.trigger.update(
                current_song_t=pos_t,
                next_slide_idx=boundary_idx,
                next_slide_t=float(self.slide_t_refs[boundary_idx]),
                slide_id=self.slide_ids[boundary_idx],
                trigger_confidence=pos_conf,
                wall_time=chunk_wall_t,
                pp_slide_index=int(self.slide_pp_indices[boundary_idx]),
            )
            if fired_now:
                triggered = True
                triggered_slide_id = self.slide_ids[boundary_idx]
                # No estimated fire time: the honest fire moment is "now"
                # (benchmark scores the chunk's file time).
            else:
                # In virtual-time mode update() may have released a pending
                # scheduled fire internally — pick it up.
                for ev in self.trigger.drain_fired():
                    triggered = True
                    triggered_slide_id = ev["slide_id"]
                    trigger_fire_t = ev["fire_at_song_t"]

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
            "dtw_cost_margin": round(dtw_result.get("cost_margin", 0.0), 4),
            "dtw_search_lo": round(dtw_result["search_lo_t"], 3),
            "dtw_search_hi": round(dtw_result["search_hi_t"], 3),
            # Anchor state
            "initialized": self._initialized,
            "obs_accepted": obs_accepted,
            "jump_pending": len(self._jump_hist) > 0,
            # HMM
            "hmm_current_slide": current_slide,
            "hmm_current_slide_id": self.slide_ids[current_slide],
            "hmm_expected_pos_t": round(hmm_out["expected_pos_t"], 3),
            "hmm_predicted_next_t": round(predicted_next_t, 3),
            "hmm_trigger_confidence": round(trigger_conf, 4),
            # Trigger
            "triggered": triggered,
            "triggered_slide_id": triggered_slide_id,
            "trigger_fire_t": round(trigger_fire_t, 3) if trigger_fire_t is not None else None,
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
