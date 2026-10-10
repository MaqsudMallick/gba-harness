"""Milestone 2: frames are the unit of time."""

from __future__ import annotations

import pytest

from gbharness.timing import (
    CYCLES_PER_FRAME,
    FRAME_MS,
    FRAME_RATE,
    GBA_CPU_HZ,
    MIN_SAFE_PRESS_FRAMES,
    frames_to_ms,
    ms_to_frames,
)


def test_gba_clock_constants():
    assert GBA_CPU_HZ == 16_777_216 == 2**24
    assert CYCLES_PER_FRAME == 280_896


def test_frame_rate_is_five_nine_point_seven_not_sixty():
    assert FRAME_RATE == GBA_CPU_HZ / CYCLES_PER_FRAME
    assert FRAME_RATE == pytest.approx(59.7275, abs=1e-4)
    assert FRAME_RATE != 60


def test_refresh_rate_matches_the_original_game_boy():
    # 4_194_304 / 70_224 is the same 59.7275. Only the clock and cycle count differ, so
    # every frame-timing conclusion from the Game Boy carries over unchanged.
    assert FRAME_RATE == pytest.approx(4_194_304 / 70_224, abs=1e-9)


def test_frame_duration():
    assert FRAME_MS == pytest.approx(16.742, abs=1e-3)


def test_minimum_safe_press_is_two_frames():
    # One frame is a coin flip against the game's once-per-frame KEYINPUT read.
    assert MIN_SAFE_PRESS_FRAMES == 2


def test_zero_is_zero():
    assert ms_to_frames(0) == 0
    assert frames_to_ms(0) == 0.0


def test_round_trip_is_stable_for_whole_frames():
    for frames in (1, 2, 3, 10, 60, 600):
        assert ms_to_frames(frames_to_ms(frames)) == frames


def test_one_second_is_about_sixty_frames():
    assert ms_to_frames(1000) == 60  # 59.7275 rounds to 60


def test_round_modes_differ_on_a_partial_frame():
    ms = FRAME_MS * 2.5
    assert ms_to_frames(ms, round_mode="down") == 2
    assert ms_to_frames(ms, round_mode="up") == 3
    assert ms_to_frames(ms, round_mode="nearest") == 3  # half rounds up, not to even


def test_nearest_rounds_half_away_from_zero_not_bankers():
    # Python's round() would give 2 for 2.5. We want 3, so a "minimum 2.5 frames" never
    # silently becomes 2.
    assert ms_to_frames(FRAME_MS * 2.5) == 3
    assert ms_to_frames(FRAME_MS * 3.5) == 4


def test_up_never_undershoots():
    for mult in (0.1, 1.0, 1.01, 2.9):
        assert ms_to_frames(FRAME_MS * mult, round_mode="up") >= mult


@pytest.mark.parametrize("bad", [-1, -0.001])
def test_negative_ms_rejected(bad):
    with pytest.raises(ValueError, match="non-negative"):
        ms_to_frames(bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_ms_rejected(bad):
    with pytest.raises(ValueError, match="finite"):
        ms_to_frames(bad)


def test_negative_frames_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        frames_to_ms(-1)


def test_unknown_round_mode_rejected():
    with pytest.raises(ValueError, match="round_mode"):
        ms_to_frames(100, round_mode="sideways")  # type: ignore[arg-type]
