"""Live ProPresenter integration tests (auto-skip when PP is unreachable).

These talk to a real ProPresenter instance on localhost:1025 (override via
PPSYNC_PP_HOST / PPSYNC_PP_PORT).  They visibly change slides and restore the
initial slide afterwards.  When ProPresenter is not running, every test here
skips, so the suite stays green offline/in CI.

Full closed-loop sweep over all slides: tools/pp_trigger_test.py.
"""

from __future__ import annotations

import os
import time

import pytest

HOST = os.environ.get("PPSYNC_PP_HOST", "localhost")
PORT = int(os.environ.get("PPSYNC_PP_PORT", "1025"))


@pytest.fixture(scope="module")
def pro():
    """Connected controller, or skip the module when PP is unreachable."""
    from propresenter_client.main import ProPresenterController

    ctrl = ProPresenterController(host=HOST, port=PORT, timeout=2)
    if ctrl.get_status() is None:
        pytest.skip(f"ProPresenter not reachable at {HOST}:{PORT}")
    return ctrl


@pytest.fixture()
def restore_slide(pro):
    """Snapshot the active slide index and restore it after the test."""
    initial = pro.get_slide_index()
    yield initial
    if initial is not None:
        pro.go_to_slide(initial + 1)
        time.sleep(0.3)


def _settle_index(pro, timeout_s: float = 2.0):
    deadline = time.monotonic() + timeout_s
    idx = pro.get_slide_index()
    while idx is None and time.monotonic() < deadline:
        time.sleep(0.1)
        idx = pro.get_slide_index()
    return idx


def test_go_to_slide_roundtrip(pro, restore_slide):
    """Commanded slide is the one ProPresenter reports active."""
    target = 0 if restore_slide != 0 else 1
    assert pro.go_to_slide(target + 1)
    time.sleep(0.6)
    assert _settle_index(pro) == target


def test_trigger_scheduler_drives_propresenter(pro, restore_slide):
    """The exact ppsync call path: TriggerScheduler._fire -> go_to_slide."""
    from ppsync.trigger import TriggerScheduler

    target = 2 if restore_slide != 2 else 3
    sched = TriggerScheduler(pp_controller=pro)
    fired = sched.update(
        current_song_t=10.0,
        next_slide_idx=5,
        next_slide_t=10.1,
        slide_id="live_test",
        trigger_confidence=0.95,
        wall_time=100.0,
        pp_slide_index=target,
    )
    assert fired
    time.sleep(0.8)  # daemon thread + PP propagation
    assert _settle_index(pro) == target
    assert sched.last_fire_result is not None
    assert sched.last_fire_result["ok"] is True
    assert sched.last_fire_result["mode"] == "propresenter"
