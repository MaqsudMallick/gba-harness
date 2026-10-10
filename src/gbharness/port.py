"""The seam: the contract every input backend implements.

Four methods -- *edges plus a clock*. Everything higher-level (taps, holds, combos,
sequences, macros, budgets) is built on top in `scheduler` and `safety`, which is what makes
those layers backend-independent and testable against a recording double.

Keeping the surface this small is deliberate. In particular there is no frame counter here:
the emulator backend has one, but a synthetic-keystroke backend driving an external emulator
has no frame concept at all, so a `frame` property would be a lie for half the
implementations. The scheduler is the only caller of `advance`, so it owns the count.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from gbharness.buttons import Button


class EmulationEnded(RuntimeError):
    """An operation was attempted on a port whose emulation has finished or been closed."""


@runtime_checkable
class InputPort(Protocol):
    """A sink for Game Boy Advance button edges, plus a way to advance time.

    Implementations must honour these guarantees, which the test suite enforces for every
    backend via a shared contract test:

    * `press` and `release` are **idempotent**. Pressing a held button or releasing an
      unheld one is a silent no-op, never an error.
    * `press` and `release` do not advance time. Nothing reaches the game until `advance`.
    * Edges applied between two `advance` calls all land in the same frame, so a combo is
      genuinely simultaneous from the game's point of view.
    * `pressed()` reflects edges already applied, including ones not yet observed by the
      game because `advance` has not been called.
    """

    def press(self, button: Button) -> None:
        """Hold `button` down. Idempotent. Does not advance time."""
        ...

    def release(self, button: Button) -> None:
        """Let `button` up. Idempotent, including for a button that was never held."""
        ...

    def advance(self, frames: int) -> bool:
        """Run the emulator forward exactly `frames` frames.

        Returns:
            True while emulation is still live, False once it has ended. Once it has
            returned False it keeps returning False without touching the emulator again.

            **`advance(0)` returns liveness, not False.** A zero-frame advance is a
            legal no-op and must never be mistaken for "the game ended". The libretro
            backend has no end-of-emulation signal at all and returns True while open;
            the recording double can simulate an ending so callers stay honest.

        Raises:
            ValueError: if `frames` is negative.
            EmulationEnded: if the port has been closed.
        """
        ...

    def pressed(self) -> frozenset[Button]:
        """The set of buttons currently held."""
        ...


@runtime_checkable
class Snapshotting(Protocol):
    """Optional capability: capture and restore emulator state.

    Separate from `InputPort` because it is genuinely optional -- an emulator backend has
    save states, a synthetic-keystroke backend driving someone else's emulator does not.
    Callers must feature-detect with `isinstance(port, Snapshotting)`.
    """

    def snapshot(self) -> bytes:
        """Capture complete emulator state."""
        ...

    def restore(self, state: bytes) -> None:
        """Restore state previously returned by `snapshot`.

        Held buttons are *not* part of the restored input state: a restore leaves the
        port's held-button set untouched, because the buttons a human or agent is holding
        are a property of the controller, not of the cartridge.
        """
        ...
