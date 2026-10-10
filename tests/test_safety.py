"""Milestone 5: every guard in the plan's table, plus the clamp-vs-reject policy."""

from __future__ import annotations

import pytest

from gbharness.actions import hold, press, repeat, sequence, tap, wait
from gbharness.buttons import Button
from gbharness.safety import (
    BudgetExhausted,
    CallTooLarge,
    Clamp,
    EpisodeBudget,
    GuardReport,
    Limits,
    OpposedInput,
    check_call_size,
    check_opposed,
    normalize,
    opposed_conflicts,
    total_frames,
)


@pytest.fixture
def report() -> GuardReport:
    return GuardReport()


# --- total_frames -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (tap("a", 2, 6), 8),
        (hold("a", 45), 45),
        (wait(30), 30),
        (press("a"), 0),
        (sequence(tap("a", 2, 6), wait(30)), 38),
        (repeat(tap("a", 2, 6), 3), 24),
        (repeat(sequence(wait(5), wait(5)), 4), 40),
    ],
)
def test_total_frames_is_exact(action, expected):
    assert total_frames(action) == expected


# --- minimum hold -------------------------------------------------------------------


@pytest.mark.parametrize("requested", [0, 1])
def test_sub_minimum_holds_are_floored_to_two(requested, report):
    result = normalize(tap("a", hold_frames=requested), Limits(), report)
    assert result.hold_frames == 2
    assert report.clamps == [Clamp("hold_frames", requested, 2)]
    assert "clamped" in report.warnings[0]


def test_zero_wait_is_not_floored(report):
    # A zero-frame wait is a legitimate no-op; a zero-frame press is not an input.
    assert normalize(wait(0), Limits(), report).frames == 0
    assert report.clamps == []


def test_zero_gap_is_not_floored(report):
    assert normalize(tap("a", 2, 0), Limits(), report).gap_frames == 0
    assert report.clamps == []


def test_hold_longer_than_max_is_clamped(report):
    limits = Limits(max_hold_frames=100)
    assert normalize(hold("a", 9999), limits, report).frames == 100
    assert report.clamps == [Clamp("frames", 9999, 100)]


def test_clamps_recurse_into_sequences_and_repeats(report):
    action = repeat(sequence(tap("a", 0), hold("b", 9999)), 2)
    normalize(action, Limits(max_hold_frames=50), report)
    assert len(report.clamps) == 2


def test_clamped_mapping_is_json_friendly(report):
    normalize(wait(99_999), Limits(max_frames_per_call=600), report)
    assert report.clamped == {"frames": {"requested": 99_999, "applied": 600}}


# --- per-call ceiling: reject, don't truncate ---------------------------------------


def test_a_huge_wait_is_clamped_then_fits(report):
    limits = Limits(max_frames_per_call=600)
    normalized = normalize(wait(9_999_999), limits, report)
    assert check_call_size(normalized, limits) == 600


def test_an_oversized_tree_is_rejected_not_truncated():
    # Clamping each value is not enough: 1000 x 60 frames each stays under every
    # per-value limit while totalling 60,000 frames.
    limits = Limits(max_frames_per_call=600)
    action = repeat(wait(60), 1000)
    with pytest.raises(CallTooLarge) as exc:
        check_call_size(action, limits)
    message = str(exc.value)
    assert "60000" in message and "600" in message
    assert "Split it" in message, "the error should name the remedy"


def test_call_size_error_quotes_seconds_of_game_time():
    with pytest.raises(CallTooLarge, match=r"10\.0s"):
        check_call_size(repeat(wait(60), 100), Limits(max_frames_per_call=600))


def test_a_call_exactly_at_the_limit_is_allowed():
    limits = Limits(max_frames_per_call=600)
    assert check_call_size(wait(600), limits) == 600


# --- opposed pairs ------------------------------------------------------------------


def test_opposed_conflicts_detects_both_axes():
    assert opposed_conflicts(frozenset({Button.UP, Button.DOWN})) == ((Button.UP, Button.DOWN),)
    both_axes = frozenset({Button.UP, Button.DOWN, Button.LEFT, Button.RIGHT})
    assert len(opposed_conflicts(both_axes)) == 2
    assert opposed_conflicts(frozenset({Button.UP, Button.LEFT})) == ()


def test_warn_policy_records_a_warning_and_continues(report):
    check_opposed(frozenset({Button.LEFT, Button.RIGHT}), Limits(), report)
    assert len(report.warnings) == 1
    assert "left+right" in report.warnings[0]
    assert "glitch" in report.warnings[0]


def test_reject_policy_raises(report):
    with pytest.raises(OpposedInput, match="up\\+down"):
        check_opposed(frozenset({Button.UP, Button.DOWN}), Limits(opposed_pairs="reject"), report)


def test_allow_policy_is_silent(report):
    check_opposed(frozenset({Button.UP, Button.DOWN}), Limits(opposed_pairs="allow"), report)
    assert report.warnings == []


def test_non_opposing_combo_is_never_flagged(report):
    check_opposed(frozenset({Button.UP, Button.A, Button.B}), Limits(), report)
    assert report.warnings == []


def test_duplicate_warnings_are_collapsed(report):
    for _ in range(5):
        check_opposed(frozenset({Button.UP, Button.DOWN}), Limits(), report)
    assert len(report.warnings) == 1


# --- episode budget ----------------------------------------------------------------


def test_unlimited_budget_never_refuses():
    budget = EpisodeBudget(None)
    budget.check(10**9)
    budget.charge(10**9)
    assert budget.remaining is None


def test_budget_tracks_and_refuses():
    budget = EpisodeBudget(100)
    budget.check(60)
    budget.charge(60)
    assert budget.remaining == 40
    budget.check(40)
    with pytest.raises(BudgetExhausted, match="60/100"):
        budget.check(41)


def test_budget_refuses_before_any_input_runs():
    # check() is called with the exact total up front, so a refusal happens before the
    # first edge rather than halfway through a sequence.
    budget = EpisodeBudget(10)
    with pytest.raises(BudgetExhausted):
        budget.check(11)
    assert budget.consumed == 0


def test_remaining_never_goes_negative():
    budget = EpisodeBudget(10)
    budget.charge(25)
    assert budget.remaining == 0
