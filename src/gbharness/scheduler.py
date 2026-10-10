"""Execute actions against an `InputPort`, frame by frame.

The hardest correctness in this package. Every bug class here is off-by-one-frame and
*silent*: the game simply misses an input now and then, which reads as flaky hardware
rather than a logic error. So the invariants are spelled out and tested directly against
recorded edges, not inferred from whether a game appears to respond.

Invariants:

* Edges within one action's press phase all land in the same frame -- a combo is genuinely
  simultaneous, never staggered across frames.
* Edge ordering within a frame is canonical (`buttons.canon_sort`), so transcripts and
  replays are byte-stable. Hardware does not care; determinism does.
* `frame` counts every frame advanced through this scheduler, monotonically. It is the
  transcript's clock and is never rewound, including by a snapshot restore.
* **`Tap` and `Hold` release only the buttons they themselves pressed.** A button already
  held when the action began stays held. This matters more than it looks: holding B to run
  while tapping A to jump, or holding R while tapping A, is the most common input pattern
  on the console, and a naive "release everything in my button set" would break it
  silently.
* If emulation ends mid-action, the remaining steps are abandoned but the release phase
  still runs, so a held button can never outlive the run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from gbharness.actions import Action, Hold, Press, Release, Repeat, Sequence, Tap, Wait
from gbharness.buttons import Button, canon_sort
from gbharness.port import InputPort


@dataclass(frozen=True, slots=True)
class RunResult:
    """What one `Scheduler.run` call did.

    Attributes:
        frames_advanced: frames consumed by this call.
        start_frame: scheduler clock before the call.
        end_frame: scheduler clock after the call.
        emulation_ended: True if the emulator reported that emulation finished during this
            call. The action may have been abandoned part-way through.
        buttons_held: buttons still held when the call returned.
    """

    frames_advanced: int
    start_frame: int
    end_frame: int
    emulation_ended: bool
    buttons_held: frozenset[Button]


@dataclass
class Scheduler:
    """Turns declarative actions into timed edges on a port.

    Holds no lock and is not thread-safe by itself -- `InputSession` owns serialization.
    """

    port: InputPort
    frame: int = 0
    guard: Callable[[frozenset[Button]], None] | None = None
    """Called with the held set immediately after every press phase.

    This is how the opposed-pair policy stays accurate: checking an action tree's button
    sets statically would over-approximate (warning about up and down that merely appear
    in the same sequence, never held together), and over-eager warnings get ignored. The
    hook sees the actual simultaneous held set, and may raise to refuse it -- the release
    phase still runs, so a refusal cannot strand a button.
    """

    _ended: bool = field(default=False, repr=False)

    def run(self, action: Action) -> RunResult:
        """Execute one action (which may be a whole `Sequence`) to completion."""
        start = self.frame
        self._execute(action)
        return RunResult(
            frames_advanced=self.frame - start,
            start_frame=start,
            end_frame=self.frame,
            emulation_ended=self._ended,
            buttons_held=self.port.pressed(),
        )

    @property
    def ended(self) -> bool:
        return self._ended

    # --- internals ------------------------------------------------------------------

    def _advance(self, frames: int) -> bool:
        """Advance the port and the clock together. Returns False once emulation ended."""
        if self._ended:
            return False
        if frames == 0:
            return True
        live = self.port.advance(frames)
        # The clock tracks frames we asked for. The port either ran them all or emulation
        # ended during them; either way this many frames of game time have gone by, and a
        # transcript that under-counts them would not replay.
        self.frame += frames
        if not live:
            self._ended = True
        return live

    def _execute(self, action: Action) -> None:
        if self._ended:
            return
        match action:
            case Press():
                ours = canon_sort(action.buttons - self.port.pressed())
                for button in ours:
                    self.port.press(button)
                if self.guard is not None:
                    try:
                        self.guard(self.port.pressed())
                    except Exception:
                        # A refused open-ended press must not stay held.
                        for button in ours:
                            self.port.release(button)
                        raise
            case Release():
                targets = action.buttons or self.port.pressed()
                for button in canon_sort(targets):
                    self.port.release(button)
            case Wait():
                self._advance(action.frames)
            case Tap():
                self._press_hold_release(action.buttons, action.hold_frames)
                self._advance(action.gap_frames)
            case Hold():
                self._press_hold_release(action.buttons, action.frames)
            case Sequence():
                for item in action.items:
                    if self._ended:
                        return
                    self._execute(item)
            case Repeat():
                for _ in range(action.times):
                    if self._ended:
                        return
                    self._execute(action.item)
            case _:
                raise TypeError(f"not an action: {action!r}")

    def _press_hold_release(self, buttons: frozenset[Button], frames: int) -> None:
        """Press, wait, release -- releasing only what this call actually pressed.

        The `finally` is the point: if `_advance` ends emulation (or anything else raises)
        the buttons this action pressed still come back up. A stuck D-pad is the one
        failure an input layer must not be capable of.
        """
        ours = canon_sort(buttons - self.port.pressed())
        for button in ours:
            self.port.press(button)
        try:
            if self.guard is not None:
                self.guard(self.port.pressed())
            self._advance(frames)
        finally:
            for button in ours:
                self.port.release(button)
