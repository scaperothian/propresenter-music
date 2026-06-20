"""TriggerScheduler: gating, ordering, and ProPresenter URL construction."""

import pytest

import ppsync.trigger as trigger_mod
from ppsync.trigger import TriggerScheduler


def _fire_kwargs(**over):
    kw = dict(
        current_song_t=10.0,
        next_slide_idx=1,
        next_slide_t=10.1,
        slide_id="01_verse1",
        trigger_confidence=0.9,
        wall_time=100.0,
        pp_slide_index=1,
    )
    kw.update(over)
    return kw


def test_fires_when_position_crosses_boundary():
    t = TriggerScheduler(dry_run=True)
    assert t.update(**_fire_kwargs())
    assert t.last_triggered_idx == 1


def test_blocked_below_confidence():
    t = TriggerScheduler(dry_run=True)
    assert not t.update(**_fire_kwargs(trigger_confidence=0.3))


def test_blocked_before_fire_window():
    t = TriggerScheduler(dry_run=True)
    assert not t.update(**_fire_kwargs(current_song_t=5.0, next_slide_t=10.0))


def test_no_refire_same_or_earlier_slide():
    t = TriggerScheduler(dry_run=True)
    assert t.update(**_fire_kwargs())
    assert not t.update(**_fire_kwargs(wall_time=200.0))  # same idx
    assert not t.update(**_fire_kwargs(next_slide_idx=0, wall_time=300.0))


def test_cooldown_blocks_rapid_consecutive_fires():
    t = TriggerScheduler(dry_run=True)
    assert t.update(**_fire_kwargs())
    assert not t.update(**_fire_kwargs(next_slide_idx=2, next_slide_t=10.2,
                                       wall_time=100.5))
    assert t.update(**_fire_kwargs(next_slide_idx=2, next_slide_t=10.2,
                                   wall_time=101.5))


def test_mark_skipped_advances_pointer():
    t = TriggerScheduler(dry_run=True)
    t.mark_skipped(3)
    assert t.last_triggered_idx == 3
    t.mark_skipped(1)  # never moves backwards
    assert t.last_triggered_idx == 3


class FakeController:
    """Stands in for propresenter_client.ProPresenterController."""

    def __init__(self):
        self.calls: list[int] = []

    def go_to_slide(self, slide_number: int) -> bool:
        self.calls.append(slide_number)
        return True


def test_pp_mode_calls_go_to_slide_with_mapped_index():
    """Repeated chorus instance (manifest idx 10) must hit pp slide 4
    — go_to_slide is 1-indexed, so it receives 5."""
    ctrl = FakeController()
    t = TriggerScheduler(pp_controller=ctrl)
    fired = t.update(**_fire_kwargs(next_slide_idx=10, slide_id="10_chorus",
                                    pp_slide_index=4))
    assert fired
    # _fire runs on a daemon thread; join via polling.
    import time
    for _ in range(100):
        if ctrl.calls:
            break
        time.sleep(0.01)
    assert ctrl.calls == [5]


def test_pp_dry_run_does_not_call_controller():
    ctrl = FakeController()
    t = TriggerScheduler(pp_controller=ctrl, dry_run=True)
    assert t.update(**_fire_kwargs(pp_slide_index=3))
    assert ctrl.calls == []


# ---------------------------------------------------------------------------
# Boundary selection (aligner.select_trigger_boundary)
# ---------------------------------------------------------------------------

import numpy as np

from ppsync.aligner import select_trigger_boundary

T_REFS = np.array([0.0, 30.0, 42.0, 54.0, 64.0, 74.0, 86.0, 95.0,
                   106.0, 117.0], dtype=np.float64)


def test_midsong_join_fires_current_slide_and_skips_older():
    """Lock-on inside slide 8 (106-117s): skip 0-7, fire 8 immediately."""
    skips, boundary = select_trigger_boundary(-1, T_REFS, 110.0)
    assert skips == list(range(0, 8))
    assert boundary == 8


def test_normal_progress_aims_at_next_boundary():
    """Mid slide 3, slide 3 already fired: aim at 4, nothing skipped."""
    skips, boundary = select_trigger_boundary(3, T_REFS, 60.0)
    assert skips == []
    assert boundary == 4


def test_jitter_step_over_boundary_still_fires_it():
    """pos jumps from 63.9 to 64.3 between chunks: slide 4 fires late."""
    skips, boundary = select_trigger_boundary(3, T_REFS, 64.3)
    assert skips == []
    assert boundary == 4


def test_cold_start_at_song_beginning_fires_first_slide():
    skips, boundary = select_trigger_boundary(-1, T_REFS, 7.0)
    assert skips == []
    assert boundary == 0


def test_position_behind_already_fired_aims_forward():
    """DTW slips backward after slide 5 fired: keep aiming at 6."""
    skips, boundary = select_trigger_boundary(5, T_REFS, 70.0)
    assert skips == []
    assert boundary == 6


# ---------------------------------------------------------------------------
# Same on-screen slide detection (aligner.same_onscreen_slide) — jump guard
# refuses jumps to another instance of the slide already shown.
# ---------------------------------------------------------------------------

from ppsync.aligner import same_onscreen_slide

# Drive-shaped: choruses (pp 4 / pp 5) repeat; verses/bridge are distinct.
PP_T_REFS = np.array([0.0, 30.0, 64.0, 74.0, 128.0, 138.0, 169.0, 194.0],
                     dtype=np.float64)
PP_IDX = np.array([0, 1, 4, 5, 4, 5, 10, 5], dtype=np.int32)
#                  v   v  ch ch ch ch  br ch   (ch5 @ 74,138,194 all pp 5)


def test_same_slide_true_for_repeated_chorus():
    # 74s and 194s both land on pp_slide_index 5 (identical chorus text)
    assert same_onscreen_slide(PP_T_REFS, PP_IDX, 80.0, 196.0) is True
    # 138s (pp 5) vs 194s (pp 5) — the live Drive 11->14 case
    assert same_onscreen_slide(PP_T_REFS, PP_IDX, 140.0, 195.0) is True


def test_different_slide_false_for_distinct_sections():
    # chorus (pp 5 @ 138) vs bridge (pp 10 @ 169) — a real progression
    assert same_onscreen_slide(PP_T_REFS, PP_IDX, 140.0, 170.0) is False
    # chorus line A (pp 4 @ 64) vs line B (pp 5 @ 74) — different text
    assert same_onscreen_slide(PP_T_REFS, PP_IDX, 66.0, 76.0) is False


def test_same_slide_handles_pre_first_boundary():
    # times before the first boundary clamp to slide 0; equal to itself
    assert same_onscreen_slide(PP_T_REFS, PP_IDX, -5.0, 10.0) is True


# ---------------------------------------------------------------------------
# Scheduled (timer-based) firing
# ---------------------------------------------------------------------------

def test_virtual_mode_fires_at_exact_scheduled_time():
    """Crossing predicted between updates fires at the timer deadline.

    In virtual mode wall_time IS file time (the benchmark clock).  With an
    unbiased estimate (pos == file time) the deadline lands at boundary -
    buffer; with a lagging estimate the deadline lags equally — modelling
    the live wall timer honestly.
    """
    t = TriggerScheduler(dry_run=True, wall_timers=False, schedule_horizon_sec=0.5)
    # boundary 10.0, buffer 0.2 -> fire_at 9.8; pos 9.7 at file time 9.7
    assert not t.update(**_fire_kwargs(current_song_t=9.7, next_slide_t=10.0,
                                       wall_time=9.7))
    assert t.last_triggered_idx == -1
    # next update at file time 9.9 releases the pending fire scheduled at 9.8
    assert not t.update(**_fire_kwargs(current_song_t=9.9, next_slide_t=10.0,
                                       wall_time=9.9))
    fired = t.drain_fired()
    assert len(fired) == 1
    assert fired[0]["fire_at_song_t"] == pytest.approx(9.8)
    assert t.last_triggered_idx == 1


def test_virtual_mode_lagging_estimate_fires_late_by_the_lag():
    """A 0.5s-lagging estimate arms the timer 0.5s late — the recorded fire
    time must show that lag, not mask it."""
    t = TriggerScheduler(dry_run=True, wall_timers=False, schedule_horizon_sec=0.5)
    # file time 10.2 but estimate says 9.7 (0.5s lag): eta 0.1 -> deadline 10.3
    assert not t.update(**_fire_kwargs(current_song_t=9.7, next_slide_t=10.0,
                                       wall_time=10.2))
    assert not t.update(**_fire_kwargs(current_song_t=9.9, next_slide_t=10.0,
                                       wall_time=10.4))
    fired = t.drain_fired()
    assert len(fired) == 1
    assert fired[0]["fire_at_song_t"] == pytest.approx(10.3)  # 9.8 + 0.5 lag


def test_virtual_mode_pending_cancelled_when_confidence_drops():
    t = TriggerScheduler(dry_run=True, wall_timers=False, schedule_horizon_sec=0.5)
    assert not t.update(**_fire_kwargs(current_song_t=9.7, next_slide_t=10.0,
                                       wall_time=100.0))
    # estimate goes unconfident before the crossing: pending must be dropped
    assert not t.update(**_fire_kwargs(current_song_t=9.75, next_slide_t=10.0,
                                       trigger_confidence=0.1, wall_time=100.05))
    assert not t.update(**_fire_kwargs(current_song_t=9.78, next_slide_t=10.0,
                                       trigger_confidence=0.1, wall_time=100.08))
    assert t.drain_fired() == []
    assert t.last_triggered_idx == -1


def test_wall_timer_fires_between_updates():
    import time as _time

    t = TriggerScheduler(dry_run=True, wall_timers=True, schedule_horizon_sec=0.5)
    assert not t.update(**_fire_kwargs(current_song_t=9.72, next_slide_t=10.0,
                                       wall_time=None))  # eta 80ms timer
    _time.sleep(0.3)
    fired = t.drain_fired()
    assert len(fired) == 1
    assert fired[0]["slide_id"] == "01_verse1"
    assert t.last_triggered_idx == 1


def test_wall_timer_rearm_cancels_stale_timer():
    import time as _time

    t = TriggerScheduler(dry_run=True, wall_timers=True, schedule_horizon_sec=0.5)
    assert not t.update(**_fire_kwargs(current_song_t=9.65, next_slide_t=10.0,
                                       wall_time=None))
    # new estimate says we're further away — re-arm with a later fire
    assert not t.update(**_fire_kwargs(current_song_t=9.62, next_slide_t=10.0,
                                       wall_time=None))
    _time.sleep(0.4)
    assert len(t.drain_fired()) == 1  # exactly one fire despite two arms
