"""Milestone 4: action value objects and their serialization."""

from __future__ import annotations

import dataclasses
import json

import pytest

from gbharness.actions import (
    DEFAULT_GAP_FRAMES,
    DEFAULT_HOLD_FRAMES,
    Hold,
    Press,
    Release,
    Sequence,
    Tap,
    Wait,
    from_dict,
    hold,
    press,
    release,
    repeat,
    sequence,
    tap,
    to_dict,
    wait,
)
from gbharness.buttons import Button


def test_defaults_are_the_safe_minimums():
    t = tap("a")
    assert t.hold_frames == DEFAULT_HOLD_FRAMES == 2
    assert t.gap_frames == DEFAULT_GAP_FRAMES


def test_convenience_constructors_parse_tokens():
    assert tap("a").buttons == frozenset({Button.A})
    assert tap(["up", "a"]).buttons == frozenset({Button.UP, Button.A})
    assert hold("left", 30).frames == 30
    assert press(["start"]).buttons == frozenset({Button.START})


def test_release_with_no_arguments_means_everything():
    assert release().buttons == frozenset()
    assert Release().buttons == frozenset()


def test_actions_are_frozen():
    t = tap("a")
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.hold_frames = 99  # type: ignore[misc]


def test_actions_are_hashable_so_they_can_be_deduplicated():
    assert len({tap("a"), tap("a"), tap("b")}) == 2
    assert hash(sequence(tap("a"), wait(3)))


@pytest.mark.parametrize(
    "ctor", [lambda: Tap(frozenset()), lambda: Hold(frozenset(), 5), lambda: Press(frozenset())]
)
def test_actions_that_need_buttons_reject_an_empty_set(ctor):
    with pytest.raises(ValueError, match="at least one button"):
        ctor()


@pytest.mark.parametrize(
    "ctor",
    [
        lambda: Tap(frozenset({Button.A}), -1),
        lambda: Tap(frozenset({Button.A}), 2, -1),
        lambda: Hold(frozenset({Button.A}), -5),
        lambda: Wait(-1),
    ],
)
def test_negative_frames_rejected_at_construction(ctor):
    with pytest.raises(ValueError, match="non-negative"):
        ctor()


def test_bool_is_not_an_acceptable_frame_count():
    # True == 1 would otherwise sail through and produce a 1-frame press, which is exactly
    # the coin-flip duration the whole package exists to avoid.
    with pytest.raises(TypeError):
        Wait(True)  # type: ignore[arg-type]


def test_repeat_requires_at_least_one():
    with pytest.raises(ValueError, match="at least 1"):
        repeat(tap("a"), 0)
    with pytest.raises(ValueError, match="at least 1"):
        repeat(tap("a"), -3)


def test_sequence_items_must_be_a_tuple_to_stay_hashable():
    with pytest.raises(TypeError, match="tuple"):
        Sequence([tap("a")])  # type: ignore[arg-type]


# --- serialization -----------------------------------------------------------------


@pytest.mark.parametrize(
    "action",
    [
        tap("a"),
        tap(["up", "a"], 4, 10),
        hold("left", 45),
        press(["b"]),
        release(),
        release(["b"]),
        wait(30),
        sequence(tap("a"), wait(30), hold("right", 12)),
        repeat(tap("a"), 3),
        repeat(sequence(tap("down"), tap("a")), 4),
    ],
)
def test_round_trip_through_dict(action):
    assert from_dict(to_dict(action)) == action


@pytest.mark.parametrize(
    "action",
    [tap("a"), sequence(tap("a"), wait(1)), repeat(sequence(tap("down"), tap("a")), 4)],
)
def test_round_trip_through_json(action):
    assert from_dict(json.loads(json.dumps(to_dict(action)))) == action


def test_buttons_serialize_in_canonical_order_for_stable_diffs():
    # frozenset iteration order is not guaranteed; a transcript has to diff cleanly.
    d = to_dict(tap(["a", "up", "start"]))
    assert d["buttons"] == ["up", "a", "start"]


def test_from_dict_rejects_unknown_type():
    with pytest.raises(ValueError, match="unknown action type"):
        from_dict({"type": "jump"})


def test_from_dict_rejects_missing_type():
    with pytest.raises(ValueError, match="no 'type' key"):
        from_dict({"buttons": ["a"]})


def test_from_dict_surfaces_a_missing_field():
    with pytest.raises(KeyError):
        from_dict({"type": "hold", "buttons": ["a"]})  # no frames
