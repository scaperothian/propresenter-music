"""TriggerScheduler: gating, ordering, and ProPresenter URL construction."""

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


def test_pp_url_uses_uuid_and_pp_index():
    t = TriggerScheduler(pp_base_url="http://pp:1025/", pp_uuid="ABC-123")
    assert (t._pp_trigger_url(4)
            == "http://pp:1025/v1/presentation/ABC-123/4/trigger")


def test_pp_url_falls_back_to_active_without_uuid():
    t = TriggerScheduler(pp_base_url="http://pp:1025")
    assert (t._pp_trigger_url(7)
            == "http://pp:1025/v1/presentation/active/7/trigger")


def test_pp_mode_sends_get_to_mapped_slide(monkeypatch):
    """Repeated chorus instance (manifest idx 10) must hit pp slide 4."""
    calls = []

    class FakeResp:
        status_code = 200

    monkeypatch.setattr(
        trigger_mod.requests, "get",
        lambda url, timeout=None: calls.append(url) or FakeResp(),
    )

    t = TriggerScheduler(pp_base_url="http://pp:1025", pp_uuid="ABC-123")
    fired = t.update(**_fire_kwargs(next_slide_idx=10, slide_id="10_chorus",
                                    pp_slide_index=4))
    assert fired
    # _fire runs the request on a daemon thread; join via polling.
    import time
    for _ in range(100):
        if calls:
            break
        time.sleep(0.01)
    assert calls == ["http://pp:1025/v1/presentation/ABC-123/4/trigger"]


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
