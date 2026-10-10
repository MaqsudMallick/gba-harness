"""The `InputPort` contract, run against every backend.

These are the guarantees `scheduler` and `safety` are written against. Running the same
cases over both backends is what keeps the recording double honest -- a `NullPort` that
drifts from the real core's behaviour would make the entire unit suite meaningless.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gbharness.backends import libretro_port
from gbharness.backends.libretro_port import (
    JOYPAD_ID,
    CoreNotFound,
    LibretroInputPort,
    find_core,
    validate_rom,
)
from gbharness.backends.null_port import NullPort
from gbharness.buttons import ALL, Button
from gbharness.port import EmulationEnded, InputPort, Snapshotting


@pytest.fixture(params=["null", "libretro"])
def port(request: pytest.FixtureRequest, test_rom_path, libretro_core):
    """Each contract test runs twice: once per backend.

    The libretro port is closed before the next test starts because a core keeps its state
    in C globals -- only one may be open per process.
    """
    if request.param == "null":
        p = NullPort()
        yield p
        p.close()
    else:
        with LibretroInputPort(test_rom_path, core=libretro_core) as p:
            p.advance(10)  # past the first frames
            yield p


def test_satisfies_the_protocol(port):
    assert isinstance(port, InputPort)
    assert isinstance(port, Snapshotting)


def test_starts_with_nothing_held(port):
    assert port.pressed() == frozenset()


def test_press_then_pressed_reflects_it(port):
    port.press(Button.A)
    assert port.pressed() == frozenset({Button.A})


def test_press_is_idempotent(port):
    port.press(Button.A)
    port.press(Button.A)
    port.press(Button.A)
    assert port.pressed() == frozenset({Button.A})


def test_release_is_idempotent_and_unheld_release_is_a_noop(port):
    port.release(Button.B)  # never held
    port.press(Button.A)
    port.release(Button.A)
    port.release(Button.A)
    assert port.pressed() == frozenset()


def test_combo_is_simultaneous(port):
    port.press(Button.UP)
    port.press(Button.R)
    assert port.pressed() == frozenset({Button.UP, Button.R})


def test_all_ten_buttons_can_be_held_at_once(port):
    for button in ALL:
        port.press(button)
    assert port.pressed() == ALL
    port.advance(2)
    assert port.pressed() == ALL


def test_edges_do_not_advance_time(port):
    before = port.pressed()
    port.press(Button.START)
    assert port.pressed() == before | {Button.START}


def test_advance_returns_true_while_live(port):
    assert port.advance(1) is True
    assert port.advance(10) is True


def test_advance_zero_reports_liveness_not_false(port):
    assert port.advance(0) is True


def test_advance_negative_is_rejected(port):
    with pytest.raises(ValueError, match="non-negative"):
        port.advance(-1)


def test_held_buttons_survive_an_advance(port):
    port.press(Button.RIGHT)
    port.advance(5)
    assert port.pressed() == frozenset({Button.RIGHT})


def test_snapshot_restore_round_trip(port):
    port.advance(5)
    state = port.snapshot()
    port.advance(20)
    port.restore(state)
    assert port.pressed() == frozenset()


def test_restore_leaves_held_buttons_alone(port):
    state = port.snapshot()
    port.press(Button.A)
    port.restore(state)
    # The controller is not part of the cartridge: a restore must not silently drop a hold.
    assert port.pressed() == frozenset({Button.A})


def test_operations_on_a_closed_port_raise():
    p = NullPort()
    p.close()
    for call in (lambda: p.press(Button.A), lambda: p.release(Button.A), lambda: p.advance(1)):
        with pytest.raises(EmulationEnded):
            call()


# --- NullPort-specific: recording fidelity -----------------------------------------


def test_null_port_records_frames_of_each_edge():
    p = NullPort()
    p.press(Button.A)
    p.advance(3)
    p.release(Button.A)
    p.advance(2)
    assert p.edges == [(0, "press", Button.A), (3, "release", Button.A)]
    assert p.frame == 5


def test_null_port_does_not_record_duplicate_edges():
    p = NullPort()
    p.press(Button.A)
    p.press(Button.A)
    p.release(Button.A)
    p.release(Button.A)
    assert len(p.edges) == 2


def test_null_port_can_simulate_emulation_ending():
    p = NullPort(ends_after=10)
    assert p.advance(9) is True
    assert p.advance(1) is False
    assert p.advance(1) is False


# --- libretro-specific: the core is the ground truth --------------------------------


def test_retro_run_is_exactly_one_frame(test_rom_path, libretro_core):
    """The whole timing model rests on this."""
    with LibretroInputPort(test_rom_path, core=libretro_core) as port:
        port.advance(17)
        assert port.frames_run == 17
        assert port.polls == 17, "one input poll per frame"


def test_the_core_polls_every_button_we_claim_to_map(test_rom_path, libretro_core):
    """Validates `JOYPAD_ID` against the core's actual behaviour.

    If one of our ids were wrong -- A as libretro id 1 rather than 8, say -- the core
    would poll 8 and we would answer a question it never asked, and the game would never
    see A. Asserting that every id we map is an id the core really polls catches that.
    """
    with LibretroInputPort(test_rom_path, core=libretro_core) as port:
        port.advance(5)
        assert port.queried_ids >= set(JOYPAD_ID.values())


def test_all_ten_gba_buttons_have_a_distinct_libretro_id():
    assert len(JOYPAD_ID) == 10
    assert len(set(JOYPAD_ID.values())) == 10
    # libretro numbers its pad after the SNES, so A and B are 8 and 0 -- taking them in
    # order would silently swap the two most-used buttons.
    assert JOYPAD_ID[Button.A] == 8
    assert JOYPAD_ID[Button.B] == 0


def test_the_core_reads_a_held_button_every_frame(test_rom_path, libretro_core):
    with LibretroInputPort(test_rom_path, core=libretro_core) as port:
        port.advance(2)
        port.press(Button.A)
        port.queried_ids.clear()
        port.advance(3)
        assert JOYPAD_ID[Button.A] in port.queried_ids


def test_a_second_port_is_refused(test_rom_path, libretro_core):
    with (
        LibretroInputPort(test_rom_path, core=libretro_core),
        pytest.raises(RuntimeError, match="already open"),
    ):
        LibretroInputPort(test_rom_path, core=libretro_core)


def test_a_port_can_be_opened_again_after_closing(test_rom_path, libretro_core):
    """Repeated init/deinit cycles in one process -- the whole suite depends on it."""
    for _ in range(3):
        with LibretroInputPort(test_rom_path, core=libretro_core) as port:
            port.advance(5)
            assert port.frames_run == 5


def test_close_releases_held_buttons(test_rom_path, libretro_core):
    port = LibretroInputPort(test_rom_path, core=libretro_core)
    port.press(Button.DOWN)
    port.press(Button.L)
    port.close()
    assert port.pressed() == frozenset()


def test_close_is_idempotent(test_rom_path, libretro_core):
    port = LibretroInputPort(test_rom_path, core=libretro_core)
    port.close()
    port.close()


def test_rom_can_be_loaded_from_bytes(test_rom_bytes, libretro_core):
    with LibretroInputPort(test_rom_bytes, core=libretro_core) as port:
        port.advance(3)
        assert port.frames_run == 3


# --- ROM validation: we check, because the core cannot be trusted to -----------------


def test_a_short_rom_is_rejected_before_the_core_sees_it():
    """`validate_rom` is not defensive politeness -- it prevents heap corruption.

    The core rejects a short buffer cleanly, but `retro_deinit` afterwards corrupts the
    process heap and the crash surfaces later in an unrelated call. Validating first means
    that path is never taken.
    """
    with pytest.raises(ValueError, match="shorter than"):
        validate_rom(b"not a gba rom")


def test_a_rom_without_the_fixed_value_is_rejected(test_rom_bytes):
    broken = bytearray(test_rom_bytes)
    broken[0xB2] = 0x00  # every cartridge that boots on hardware carries 0x96 here
    with pytest.raises(ValueError, match="fixed value"):
        validate_rom(bytes(broken))


def test_the_synthesized_test_rom_validates(test_rom_bytes):
    validate_rom(test_rom_bytes)


def test_the_port_validates_on_construction(libretro_core):
    with pytest.raises(ValueError, match="Game Boy Advance ROM"):
        LibretroInputPort(b"nope", core=libretro_core)


def test_the_core_still_works_after_a_rejected_rom(test_rom_bytes, libretro_core):
    """A refused ROM must leave the core usable, not poisoned."""
    with pytest.raises(ValueError):
        LibretroInputPort(b"nope", core=libretro_core)
    with LibretroInputPort(test_rom_bytes, core=libretro_core) as port:
        port.advance(5)
        assert port.frames_run == 5


# --- core discovery: one path, used by everything ------------------------------------


def test_vendor_dir_is_found_from_the_project_root(tmp_path, monkeypatch):
    """`scripts/fetch-core.sh` puts the core here, so discovery must look here.

    Regression: this search lived only in the test fixtures for a while, so `vendor/`
    worked under pytest while `examples/demo.py` reported "could not find
    mgba_libretro.so" with the core sitting right there.
    """
    vendor = tmp_path / "vendor" / "libretro"
    vendor.mkdir(parents=True)
    core = vendor / "mgba_libretro.so"
    core.write_bytes(b"\x7fELF not really")
    monkeypatch.delenv("GBHARNESS_LIBRETRO_CORE", raising=False)
    monkeypatch.chdir(tmp_path)
    assert find_core() == core


def test_vendor_dir_is_found_from_a_subdirectory(tmp_path, monkeypatch):
    vendor = tmp_path / "vendor" / "libretro"
    vendor.mkdir(parents=True)
    (vendor / "mgba_libretro.so").write_bytes(b"\x7fELF")
    deep = tmp_path / "examples" / "nested"
    deep.mkdir(parents=True)
    monkeypatch.delenv("GBHARNESS_LIBRETRO_CORE", raising=False)
    monkeypatch.chdir(deep)
    assert find_core() == vendor / "mgba_libretro.so"


def test_explicit_path_wins_over_everything(tmp_path, libretro_core):
    assert find_core(libretro_core) == Path(libretro_core)


def test_env_var_is_honoured(tmp_path, monkeypatch, libretro_core):
    monkeypatch.setenv("GBHARNESS_LIBRETRO_CORE", str(libretro_core))
    monkeypatch.chdir(tmp_path)
    assert find_core() == Path(libretro_core)


def test_missing_core_error_names_every_way_to_get_one(tmp_path, monkeypatch):
    monkeypatch.delenv("GBHARNESS_LIBRETRO_CORE", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(libretro_port, "_SYSTEM_CORE_DIRS", ())
    monkeypatch.setattr(libretro_port, "_vendor_dirs", lambda: [])
    with pytest.raises(CoreNotFound) as exc:
        find_core()
    message = str(exc.value)
    assert "fetch-core.sh" in message
    assert "apt install libretro-mgba" in message
    assert "GBHARNESS_LIBRETRO_CORE" in message
