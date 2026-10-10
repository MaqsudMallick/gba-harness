"""Milestone 4: frame-exact edge emission.

Every assertion here is on *recorded edges and their frame numbers*, not on game state.
That is deliberate: a scheduler that staggers a combo across two frames, or drops the gap
after a tap, still "works" by any end-state check and still makes the game miss inputs.
"""

from __future__ import annotations

import pytest

from gbharness.actions import hold, press, release, repeat, sequence, tap, wait
from gbharness.backends.null_port import NullPort
from gbharness.buttons import Button
from gbharness.scheduler import Scheduler


@pytest.fixture
def port() -> NullPort:
    return NullPort()


@pytest.fixture
def sched(port: NullPort) -> Scheduler:
    return Scheduler(port)


# --- the basic shapes ---------------------------------------------------------------


def test_tap_presses_holds_releases_then_gaps(sched, port):
    result = sched.run(tap("a", hold_frames=2, gap_frames=6))
    assert port.edges == [(0, "press", Button.A), (2, "release", Button.A)]
    assert result.frames_advanced == 8  # 2 held + 6 gap
    assert sched.frame == 8
    assert port.pressed() == frozenset()


def test_hold_has_no_trailing_gap(sched, port):
    result = sched.run(hold("left", 45))
    assert port.edges == [(0, "press", Button.LEFT), (45, "release", Button.LEFT)]
    assert result.frames_advanced == 45


def test_wait_emits_no_edges(sched, port):
    result = sched.run(wait(30))
    assert port.edges == []
    assert result.frames_advanced == 30
    assert sched.frame == 30


def test_wait_zero_is_a_legal_noop(sched, port):
    result = sched.run(wait(0))
    assert result.frames_advanced == 0
    assert result.emulation_ended is False
    assert port.events == []


def test_press_does_not_advance_time(sched, port):
    result = sched.run(press("b"))
    assert port.edges == [(0, "press", Button.B)]
    assert result.frames_advanced == 0
    assert port.pressed() == frozenset({Button.B})


def test_release_with_empty_set_releases_everything(sched, port):
    sched.run(press(["b", "up"]))
    sched.run(release())
    assert port.pressed() == frozenset()
    assert [e[1] for e in port.edges] == ["press", "press", "release", "release"]


def test_release_of_an_unheld_button_emits_nothing(sched, port):
    sched.run(release(["a"]))
    assert port.edges == []


# --- combos: the simultaneity invariant ---------------------------------------------


def test_combo_presses_all_land_in_the_same_frame(sched, port):
    sched.run(tap(["up", "a"], hold_frames=3, gap_frames=0))
    press_frames = {f for f, op, _ in port.edges if op == "press"}
    release_frames = {f for f, op, _ in port.edges if op == "release"}
    assert press_frames == {0}, "a combo must not be staggered across frames"
    assert release_frames == {3}


def test_combo_edge_order_is_canonical_not_set_order(sched, port):
    sched.run(tap(["a", "start", "up"], hold_frames=1, gap_frames=0))
    pressed_order = [b for _, op, b in port.edges if op == "press"]
    assert pressed_order == [Button.UP, Button.A, Button.START]


def test_soft_reset_combo_is_four_simultaneous_buttons(sched, port):
    sched.run(hold(["a", "b", "start", "select"], 10))
    assert {f for f, op, _ in port.edges if op == "press"} == {0}
    assert len([e for e in port.edges if e[1] == "press"]) == 4


# --- the run-and-jump invariant -----------------------------------------------------


def test_tap_does_not_release_a_button_held_by_an_earlier_press(sched, port):
    """Hold B to run, tap A to jump, keep running.

    The naive implementation releases every button in the action's set, which silently
    drops the run. This is the most common input pattern on the console.
    """
    sched.run(press("b"))
    sched.run(tap("a"))
    assert port.pressed() == frozenset({Button.B})
    assert port.edges == [
        (0, "press", Button.B),
        (0, "press", Button.A),
        (2, "release", Button.A),
    ]


def test_tap_overlapping_a_held_button_leaves_it_held(sched, port):
    sched.run(press("a"))
    sched.run(tap(["a", "b"]))
    # A was already ours-held, so only B gets released by the tap.
    assert port.pressed() == frozenset({Button.A})


def test_hold_restores_the_prior_held_set(sched, port):
    sched.run(press("right"))
    before = port.pressed()
    sched.run(hold(["a"], 10))
    assert port.pressed() == before


# --- composition --------------------------------------------------------------------


def test_repeat_spaces_taps_by_the_gap(sched, port):
    sched.run(repeat(tap("a", hold_frames=2, gap_frames=6), 3))
    assert port.edges == [
        (0, "press", Button.A),
        (2, "release", Button.A),
        (8, "press", Button.A),
        (10, "release", Button.A),
        (16, "press", Button.A),
        (18, "release", Button.A),
    ]
    assert sched.frame == 24


def test_zero_gap_taps_are_adjacent_and_still_distinct_edges(sched, port):
    sched.run(repeat(tap("a", hold_frames=2, gap_frames=0), 2))
    assert port.edges == [
        (0, "press", Button.A),
        (2, "release", Button.A),
        (2, "press", Button.A),
        (4, "release", Button.A),
    ]


def test_sequence_runs_in_order_and_sums_frames(sched, port):
    result = sched.run(sequence(tap("a", 2, 6), wait(30), hold("right", 12)))
    assert result.frames_advanced == 8 + 30 + 12
    assert sched.frame == 50


def test_nested_repeat_of_a_sequence(sched, port):
    sched.run(repeat(sequence(tap("down", 2, 4), tap("a", 2, 4)), 2))
    assert [f for f, op, _ in port.edges if op == "press"] == [0, 6, 12, 18]


def test_scheduler_clock_is_monotonic_across_calls(sched, port):
    sched.run(wait(10))
    sched.run(tap("a", 2, 0))
    sched.run(wait(5))
    assert sched.frame == 17
    assert port.frame == sched.frame


# --- emulation ending mid-action ----------------------------------------------------


def test_emulation_ending_mid_sequence_abandons_the_rest(port):
    sched = Scheduler(NullPort(ends_after=10))
    result = sched.run(repeat(tap("a", hold_frames=2, gap_frames=6), 5))
    assert result.emulation_ended is True
    assert result.frames_advanced == 10
    assert sched.ended is True


def test_nothing_is_left_held_when_emulation_ends_mid_hold():
    p = NullPort(ends_after=10)
    sched = Scheduler(p)
    result = sched.run(hold(["a", "up"], 500))
    assert result.emulation_ended is True
    assert p.pressed() == frozenset(), "a stuck button must be impossible"
    assert result.buttons_held == frozenset()


def test_once_ended_further_runs_are_noops():
    p = NullPort(ends_after=5)
    sched = Scheduler(p)
    sched.run(wait(10))
    edges_before = list(p.events)
    result = sched.run(tap("a"))
    assert result.frames_advanced == 0
    assert p.events == edges_before


def test_release_still_happens_if_advance_raises():
    class Exploding(NullPort):
        def advance(self, frames: int) -> bool:
            raise RuntimeError("emulator exploded")

    p = Exploding()
    sched = Scheduler(p)
    with pytest.raises(RuntimeError):
        sched.run(hold(["a", "b"], 10))
    assert p.pressed() == frozenset(), "buttons must come up even on an exception"


def test_unknown_action_type_is_rejected(sched):
    with pytest.raises(TypeError, match="not an action"):
        sched.run("press a")  # type: ignore[arg-type]
