"""Milestone 5: single-writer discipline, guard integration, and lifecycle."""

from __future__ import annotations

import threading
import time

import pytest

from gbharness.actions import hold, press, release, repeat, sequence, tap, wait
from gbharness.backends.libretro_port import JOYPAD_ID, LibretroInputPort
from gbharness.backends.null_port import NullPort
from gbharness.buttons import Button
from gbharness.port import EmulationEnded
from gbharness.safety import BudgetExhausted, CallTooLarge, Limits, OpposedInput
from gbharness.session import InputSession


@pytest.fixture
def session() -> InputSession:
    with InputSession(NullPort()) as s:
        yield s


# --- result payload -----------------------------------------------------------------


def test_result_reports_frames_and_clock(session):
    result = session.submit(tap("a", 2, 6))
    assert result.frames_advanced == 8
    assert result.total_frames == 8
    assert session.frame == 8
    assert result.emulation_ended is False


def test_result_reports_held_buttons_in_canonical_order(session):
    result = session.submit(press(["start", "up"]))
    assert result.buttons_held == (Button.UP, Button.START)


def test_result_surfaces_clamps_to_the_caller(session):
    result = session.submit(tap("a", hold_frames=0))
    # Silent clamping produces an agent that never learns. It must come back.
    assert result.clamped == {"hold_frames": {"requested": 0, "applied": 2}}
    assert any("clamped" in w for w in result.warnings)


def test_result_is_json_serializable(session):
    import json

    payload = session.submit(tap("a")).to_dict()
    assert json.loads(json.dumps(payload))["frames_advanced"] == 8


def test_result_carries_no_game_state(session):
    # The observation layer is a separate concern; conflating them is the trap.
    payload = session.submit(tap("a")).to_dict()
    assert not {"screen", "memory", "image", "pixels"} & payload.keys()


# --- guards, integrated -------------------------------------------------------------


def test_opposed_pair_warning_reaches_the_caller(session):
    result = session.submit(hold(["left", "right"], 5))
    assert any("left+right" in w for w in result.warnings)


def test_opposed_pair_warning_is_not_raised_for_a_merely_sequential_pair(session):
    # up and down appear in the same call but are never held together. An over-eager
    # static check would warn here, and warnings that cry wolf get ignored.
    result = session.submit(sequence(tap("up"), tap("down")))
    assert result.warnings == ()


def test_reject_policy_refuses_and_strands_nothing():
    with InputSession(NullPort(), Limits(opposed_pairs="reject")) as s:
        s.submit(press("left"))
        with pytest.raises(OpposedInput):
            s.submit(hold(["right"], 5))
        assert s.pressed() == frozenset({Button.LEFT}), "the refused press must not stick"


def test_oversized_call_is_refused_before_any_input(session):
    with pytest.raises(CallTooLarge):
        session.submit(repeat(wait(60), 1000))
    assert session.frame == 0, "nothing should have run"


def test_budget_is_charged_and_refuses_when_spent():
    with InputSession(NullPort(), Limits(episode_frame_budget=20)) as s:
        assert s.submit(wait(15)).budget_remaining == 5
        with pytest.raises(BudgetExhausted):
            s.submit(wait(10))
        assert s.frame == 15


def test_budget_remaining_is_none_when_unlimited(session):
    assert session.submit(wait(1)).budget_remaining is None


# --- single-writer discipline -------------------------------------------------------


class ConcurrencyDetectingPort(NullPort):
    """Flags any overlapping entry into `advance`."""

    def __init__(self) -> None:
        super().__init__()
        self.inside = False
        self.violations = 0

    def advance(self, frames: int) -> bool:
        if self.inside:
            self.violations += 1
        self.inside = True
        try:
            time.sleep(0.001)  # widen the window a real race would use
            return super().advance(frames)
        finally:
            self.inside = False


def test_concurrent_submits_never_overlap_in_the_emulator():
    port = ConcurrencyDetectingPort()
    session = InputSession(port)
    errors: list[BaseException] = []

    def worker(button: str) -> None:
        try:
            for _ in range(5):
                session.submit(tap(button, 2, 2))
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(b,))
        for b in ("a", "b", "up", "down", "left", "right", "start", "select")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert port.violations == 0, "two callers were inside the emulator at once"
    assert port.pressed() == frozenset()
    session.close()


def test_a_sequence_is_atomic_against_other_callers():
    """One caller's combo must never absorb another caller's button."""
    port = NullPort()
    session = InputSession(port)
    barrier = threading.Barrier(2)

    def hold_b() -> None:
        barrier.wait()
        session.submit(hold("b", 20))

    def tap_a() -> None:
        barrier.wait()
        session.submit(tap("a", 2, 2))

    threads = [threading.Thread(target=hold_b), threading.Thread(target=tap_a)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Whichever ran first, its press and release bracket the other's entirely.
    presses = [i for i, (_, op, _) in enumerate(port.edges) if op == "press"]
    releases = [i for i, (_, op, _) in enumerate(port.edges) if op == "release"]
    assert sorted(presses + releases) == list(range(4))
    first_release = min(releases)
    assert first_release == 1, f"edges interleaved: {port.edges}"
    session.close()


# --- lifecycle ----------------------------------------------------------------------


def test_release_all_releases_everything(session):
    session.submit(press(["a", "up"]))
    result = session.release_all()
    assert result is not None
    assert session.pressed() == frozenset()


def test_release_all_is_none_when_nothing_held(session):
    assert session.release_all() is None


def test_context_manager_releases_on_exception():
    port = NullPort()
    with pytest.raises(RuntimeError), InputSession(port) as s:
        s.submit(press(["left", "a"]))
        raise RuntimeError("agent crashed")
    assert port.pressed() == frozenset(), "a crash must not jam the D-pad"


def test_close_is_idempotent(session):
    session.close()
    session.close()
    assert session.closed


def test_a_closed_session_refuses_work():
    s = InputSession(NullPort())
    s.close()
    with pytest.raises(EmulationEnded, match="closed"):
        s.submit(tap("a"))


# --- snapshots ----------------------------------------------------------------------


def test_snapshot_restore_round_trip(session):
    session.submit(wait(5))
    state = session.snapshot()
    session.submit(wait(10))
    session.restore(state)
    assert session.pressed() == frozenset()


def test_restore_does_not_rewind_the_session_clock(session):
    state = session.snapshot()
    session.submit(wait(10))
    before = session.frame
    session.restore(state)
    # Game time went back; session time did not. Two moments must never share a frame
    # number, or the transcript cannot order them.
    assert session.frame == before


def test_snapshot_on_a_backend_without_support_is_a_clear_error():
    class Bare:
        def press(self, button): ...
        def release(self, button): ...
        def advance(self, frames):
            return True

        def pressed(self):
            return frozenset()

    s = InputSession(Bare())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="does not support snapshots"):
        s.snapshot()


# --- end to end on a real emulator --------------------------------------------------


def test_session_drives_a_real_core(test_rom_path, libretro_core):
    with (
        LibretroInputPort(test_rom_path, core=libretro_core) as port,
        InputSession(port) as session,
    ):
        session.submit(wait(10))
        session.submit(press("a"))
        session.submit(wait(2))
        assert session.pressed() == frozenset({Button.A})
        session.submit(release())
        result = session.submit(repeat(tap("start", 3, 5), 4))
        assert result.frames_advanced == 32
        assert session.frame == 10 + 2 + 32
        # Every frame the session advanced is a frame the core actually ran.
        assert port.frames_run == session.frame
        assert port.polls == session.frame


def test_the_core_only_reads_input_when_time_advances(test_rom_path, libretro_core):
    """libretro pulls the held set once per frame, so no frame means no read.

    This is the GBA equivalent of the Game Boy's "the register does not change until a
    frame is emulated", and it is why every duration here is measured in frames.
    """
    with (
        LibretroInputPort(test_rom_path, core=libretro_core) as port,
        InputSession(port) as session,
    ):
        session.submit(wait(2))
        polls_before = port.polls
        session.submit(press("a"))  # Press advances no frames
        assert port.polls == polls_before, "no frame ran, so nothing was read"
        assert session.pressed() == frozenset({Button.A}), "but we are holding it"
        session.submit(wait(1))
        assert port.polls == polls_before + 1


def test_run_and_jump_on_a_real_core(test_rom_path, libretro_core):
    """Hold B to run, tap A to jump, keep running."""
    with (
        LibretroInputPort(test_rom_path, core=libretro_core) as port,
        InputSession(port) as session,
    ):
        session.submit(wait(10))
        session.submit(press("b"))
        session.submit(tap("a", 2, 2))
        assert session.pressed() == frozenset({Button.B})


def test_shoulder_and_face_combo_on_a_real_core(test_rom_path, libretro_core):
    """Hold R while tapping A -- the GBA pattern the Game Boy could not express."""
    with (
        LibretroInputPort(test_rom_path, core=libretro_core) as port,
        InputSession(port) as session,
    ):
        session.submit(wait(10))
        session.submit(press("r"))
        port.queried_ids.clear()
        result = session.submit(tap("a", 3, 3))
        assert result.buttons_held == (Button.R,)
        # The core polled both while they were held together.
        assert {JOYPAD_ID[Button.A], JOYPAD_ID[Button.R]} <= port.queried_ids
        assert result.warnings == (), "L/R and face buttons are not an opposed pair"


def test_l_plus_r_chord_is_not_warned_about(test_rom_path, libretro_core):
    with (
        LibretroInputPort(test_rom_path, core=libretro_core) as port,
        InputSession(port) as session,
    ):
        result = session.submit(hold(["l", "r"], 4))
        assert result.warnings == ()


def test_all_ten_buttons_at_once_on_a_real_core(test_rom_path, libretro_core):
    with (
        LibretroInputPort(test_rom_path, core=libretro_core) as port,
        InputSession(port) as session,
    ):
        session.submit(wait(10))
        port.queried_ids.clear()
        result = session.submit(
            hold(["up", "down", "left", "right", "a", "b", "l", "r", "start", "select"], 4)
        )
        # Opposing D-pad axes are held here, so both warnings must surface.
        assert len(result.warnings) == 1
        assert "up+down" in result.warnings[0] and "left+right" in result.warnings[0]
        assert set(JOYPAD_ID.values()) <= port.queried_ids
