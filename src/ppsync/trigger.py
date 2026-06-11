"""Trigger scheduler: fires REST calls to the presentation software.

Applies a configurable look-ahead buffer so the slide advances slightly before
the musical boundary, accounting for display latency.  Gates on an HMM
trigger-confidence threshold to prevent false positives.

Two output modes:
  - ProPresenter (``pp_controller`` set): drives slides through a
    ``propresenter_client.ProPresenterController`` — ``go_to_slide(n)``
    (1-indexed) against the ACTIVE presentation.
  - Legacy (default): POST a JSON payload to ``rest_url``.

Requests run on a daemon thread: a slow presentation host must never stall
the 200ms real-time audio loop.

Scheduled (timer-based) firing
------------------------------
Position estimates arrive once per chunk (200ms) and ~processing-latency
late, so waiting to OBSERVE the boundary crossing fires 100ms late on
average plus the processing delay.  Instead, when a crossing is predicted
within the scheduling horizon, a fire is scheduled at the exact predicted
moment (playback advances at 1.0 song-sec per wall-sec):

  - wall_timers=True (live): a threading.Timer fires at the predicted
    moment; every newer estimate re-arms it (generation-guarded).
  - wall_timers=False (offline benchmark, faster than real time): the
    pending fire is released by the next update() whose song time passes
    the scheduled moment, and is reported as firing at that exact song
    time — modelling what the wall timer would have done live.
"""

from __future__ import annotations

import threading
import time

import requests

from .config import REST_TIMEOUT_SEC, REST_URL, TRIGGER_BUFFER_MS, TRIGGER_CONFIDENCE_MIN


class TriggerScheduler:
    """
    Decides when to fire a REST trigger for the next slide.

    State:
        _last_triggered_idx:  slide index of the most recently triggered slide
        _last_trigger_t:      wall-clock time of last trigger (prevents double-fire)
        _pending:             (slide_idx, fire_at_song_t)  queued trigger
    """

    def __init__(
        self,
        rest_url: str = REST_URL,
        buffer_ms: float = TRIGGER_BUFFER_MS,
        confidence_min: float = TRIGGER_CONFIDENCE_MIN,
        rest_timeout_sec: float = REST_TIMEOUT_SEC,
        dry_run: bool = False,
        pp_controller=None,
        wall_timers: bool = True,
        schedule_horizon_sec: float = 0.5,
    ) -> None:
        self.rest_url = rest_url
        self.buffer_sec = buffer_ms / 1000.0
        self.confidence_min = confidence_min
        self.rest_timeout_sec = rest_timeout_sec
        self.dry_run = dry_run
        self.pp_controller = pp_controller
        self.wall_timers = wall_timers
        self.schedule_horizon_sec = schedule_horizon_sec
        # Delivery observability: outcome of the most recent fire, written by
        # the sender thread and surfaced in telemetry frames.
        self.mode = ("dry-run" if dry_run
                     else "propresenter" if pp_controller is not None
                     else "legacy-post")
        self.last_fire_result: dict | None = None

        self._last_triggered_idx: int = -1
        self._last_trigger_wall_t: float = 0.0
        self._cooldown_sec: float = 1.0  # minimum seconds between triggers

        # Scheduled-fire state (guarded by _lock; the timer runs on its own
        # thread).  _pending holds the armed fire; _generation invalidates
        # stale timers after re-arms/cancels; _fired_async collects timer
        # fires for the aligner to surface in the next telemetry frame.
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._pending: dict | None = None
        self._generation: int = 0
        self._fired_async: list[dict] = []

    @property
    def last_triggered_idx(self) -> int:
        """Index of the most recently triggered (or skipped) slide."""
        return self._last_triggered_idx

    def mark_skipped(self, slide_idx: int) -> None:
        """Mark *slide_idx* as passed without firing (stale boundary)."""
        with self._lock:
            self._last_triggered_idx = max(self._last_triggered_idx, slide_idx)

    def update(
        self,
        current_song_t: float,
        next_slide_idx: int,
        next_slide_t: float,
        slide_id: str,
        trigger_confidence: float,
        wall_time: float | None = None,
        pp_slide_index: int | None = None,
    ) -> bool:
        """
        Check whether to fire the trigger for *next_slide_idx* now.

        The trigger fires when:
          1. confidence >= confidence_min
          2. next_slide_idx has not already been triggered
          3. (next_slide_t - buffer_sec) <= current_song_t
          4. wall-clock cooldown has elapsed

        Args:
            current_song_t:    current HMM-estimated song position (seconds)
            next_slide_idx:    index of the upcoming slide
            next_slide_t:      TRUE musical timestamp of the next slide boundary
            slide_id:          slide identifier string (for the REST payload)
            trigger_confidence: HMM trigger confidence [0, 1]
            wall_time:         current wall time (default: time.monotonic())

        Returns:
            True if a trigger was fired immediately by this call.  Fires that
            happen between calls (scheduled timers) are reported through
            drain_fired().
        """
        if wall_time is None:
            wall_time = time.monotonic()

        with self._lock:
            # Virtual-time mode: release a pending scheduled fire whose wall
            # deadline has passed (benchmark wall_time IS file time, so this
            # models the live timer exactly — estimate lag included: the
            # timer was armed eta seconds before the PREDICTED crossing, and
            # if the prediction lagged truth, so does the fire).
            if (not self.wall_timers and self._pending is not None
                    and wall_time >= self._pending["scheduled_wall_t"]):
                self._release_pending_locked(wall_time)

            if next_slide_idx <= self._last_triggered_idx:
                self._cancel_pending_locked()
                return False  # already triggered this or an earlier slide

            if trigger_confidence < self.confidence_min:
                self._cancel_pending_locked()  # stale prediction — don't fire blind
                return False

            fire_at = next_slide_t - self.buffer_sec
            eta = fire_at - current_song_t

            if eta <= 0:
                # Crossing already happened — fire immediately.
                self._cancel_pending_locked()
                if wall_time - self._last_trigger_wall_t < self._cooldown_sec:
                    return False
                self._last_triggered_idx = next_slide_idx
                self._last_trigger_wall_t = wall_time
            elif eta <= self.schedule_horizon_sec:
                # Crossing predicted before the next estimate: schedule the
                # fire at the exact moment (re-arming any previous schedule).
                self._arm_pending_locked(
                    next_slide_idx, slide_id, trigger_confidence,
                    fire_at, next_slide_t, pp_slide_index, eta, wall_time,
                )
                return False
            else:
                self._cancel_pending_locked()
                return False

        self._fire(next_slide_idx, slide_id, trigger_confidence, current_song_t,
                   next_slide_t, pp_slide_index)
        return True

    def drain_fired(self) -> list[dict]:
        """Return (and clear) fires performed by scheduled timers."""
        with self._lock:
            fired, self._fired_async = self._fired_async, []
            return fired

    # -- scheduled-fire internals (call with self._lock held) ---------------

    def _arm_pending_locked(self, slide_idx, slide_id, confidence, fire_at_song_t,
                            boundary_t, pp_slide_index, eta, wall_time) -> None:
        self._generation += 1
        gen = self._generation
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._pending = {
            "slide_idx": slide_idx, "slide_id": slide_id, "confidence": confidence,
            "fire_at_song_t": fire_at_song_t, "boundary_t": boundary_t,
            "pp_slide_index": pp_slide_index, "scheduled_wall_t": wall_time + eta,
        }
        if self.wall_timers:
            self._timer = threading.Timer(eta, self._timer_fire, args=(gen,))
            self._timer.daemon = True
            self._timer.start()

    def _cancel_pending_locked(self) -> None:
        self._generation += 1
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        self._pending = None

    def _timer_fire(self, generation: int) -> None:
        with self._lock:
            if generation != self._generation or self._pending is None:
                return  # re-armed or cancelled since this timer was set
            self._release_pending_locked(time.monotonic())

    def _release_pending_locked(self, wall_time: float) -> None:
        p, self._pending = self._pending, None
        self._timer = None
        if p is None or p["slide_idx"] <= self._last_triggered_idx:
            return
        if wall_time - self._last_trigger_wall_t < self._cooldown_sec:
            return
        self._last_triggered_idx = p["slide_idx"]
        self._last_trigger_wall_t = wall_time
        # In virtual mode wall time IS file time — report the timer deadline
        # as the fire moment (honest: includes any estimate lag).  In wall
        # mode the reference-time target is the only song time we know.
        fire_t = (p["scheduled_wall_t"] if not self.wall_timers
                  else p["fire_at_song_t"])
        self._fired_async.append(
            {"slide_id": p["slide_id"], "slide_idx": p["slide_idx"],
             "fire_at_song_t": fire_t}
        )
        self._fire(p["slide_idx"], p["slide_id"], p["confidence"],
                   p["fire_at_song_t"], p["boundary_t"], p["pp_slide_index"])

    def reset(self) -> None:
        """Call when a new song begins or manual resync is needed."""
        with self._lock:
            self._cancel_pending_locked()
            self._fired_async.clear()
            self._last_triggered_idx = -1
            self._last_trigger_wall_t = 0.0

    def _fire(
        self,
        slide_idx: int,
        slide_id: str,
        confidence: float,
        current_t: float,
        boundary_t: float,
        pp_slide_index: int | None = None,
    ) -> None:
        if self.pp_controller is not None and pp_slide_index is not None:
            if self.dry_run:
                print(f"[TRIGGER dry-run] {slide_id}  →  "
                      f"go_to_slide({pp_slide_index + 1})")
                self.last_fire_result = {"slide_id": slide_id, "mode": "dry-run",
                                         "ok": True}
                return
            threading.Thread(
                target=self._send_pp,
                args=(pp_slide_index, slide_id, boundary_t, confidence),
                daemon=True,
            ).start()
            return

        payload = {
            "slide_id": slide_id,
            "slide_idx": slide_idx,
            "timestamp": boundary_t,
            "confidence": round(confidence, 4),
            "current_t": round(current_t, 4),
        }
        if self.dry_run:
            print(f"[TRIGGER dry-run] {payload}")
            return
        threading.Thread(
            target=self._send_legacy, args=(payload, slide_id, boundary_t, confidence),
            daemon=True,
        ).start()

    def _send_pp(self, pp_slide_index: int, slide_id: str,
                 boundary_t: float, confidence: float) -> None:
        try:
            # go_to_slide is 1-indexed (propresenter-client convention).
            ok = self.pp_controller.go_to_slide(pp_slide_index + 1)
            print(f"[TRIGGER] slide={slide_id!r}  t={boundary_t:.2f}s  "
                  f"conf={confidence:.2f}  → go_to_slide({pp_slide_index + 1}) "
                  f"{'ok' if ok else 'FAILED'}")
            self.last_fire_result = {"slide_id": slide_id, "mode": "propresenter",
                                     "ok": bool(ok)}
        except Exception as exc:  # client raises requests exceptions internally
            print(f"[TRIGGER error] go_to_slide({pp_slide_index + 1}): {exc}")
            self.last_fire_result = {"slide_id": slide_id, "mode": "propresenter",
                                     "ok": False, "error": str(exc)}

    def _send_legacy(self, payload: dict, slide_id: str, boundary_t: float,
                     confidence: float) -> None:
        try:
            resp = requests.post(self.rest_url, json=payload, timeout=self.rest_timeout_sec)
            print(f"[TRIGGER] slide={slide_id!r}  t={boundary_t:.2f}s  "
                  f"conf={confidence:.2f}  → HTTP {resp.status_code}")
            self.last_fire_result = {"slide_id": slide_id, "mode": "legacy-post",
                                     "ok": resp.status_code < 400}
        except requests.RequestException as exc:
            print(f"[TRIGGER error] {exc}")
            self.last_fire_result = {"slide_id": slide_id, "mode": "legacy-post",
                                     "ok": False, "error": str(exc)}
