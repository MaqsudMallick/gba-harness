"""The ten Game Boy Advance inputs, and nothing else.

The GBA has exactly ten digital inputs: four D-pad directions, two face buttons (A and B),
two shoulder buttons (L and R), and Start and Select. There is no X or Y -- those are SNES
face buttons -- and no analog stick.

WASD / ZXCV is not a button set. It is one emulator's *keyboard binding* onto these ten
inputs, and every emulator chooses differently (mGBA's desktop defaults are arrows for the
D-pad, X and Z for A and B, and A and S for L and R). So keyboard names are not the
vocabulary of this package: the canonical hardware names are. Keyboard dialects exist in
`DIALECTS` as a convenience for human-facing tooling and are opt-in per call, never wired
into the agent-facing tool schemas.

Two collisions make that separation worth enforcing rather than merely preferring::

    parse("a")                  # -> Button.A      the A face button
    parse("a", dialect="wasd")  # -> Button.LEFT   the left D-pad direction
    parse("a", dialect="mgba")  # -> Button.L      the left *shoulder* button

    parse("l")                  # -> Button.L      the shoulder, NOT the D-pad
    parse("left")               # -> Button.LEFT   the D-pad direction

Every reading is correct in its own dialect, which is exactly why a dialect always has to
be stated. And in canonical names, ``l``/``r`` are the shoulders while ``left``/``right``
are directions -- the one abbreviation this package will not accept.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class Button(StrEnum):
    """A Game Boy Advance hardware input.

    Declaration order is the canonical order used wherever a deterministic ordering of a
    button set is needed (press/release sequencing, transcript output, test assertions).
    """

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    A = "a"
    B = "b"
    L = "l"
    R = "r"
    START = "start"
    SELECT = "select"


CANON_ORDER: Final[tuple[Button, ...]] = tuple(Button)
"""Every button in canonical (declaration) order."""

_CANON_INDEX: Final[dict[Button, int]] = {b: i for i, b in enumerate(CANON_ORDER)}

ALL: Final = frozenset(Button)
DPAD: Final = frozenset({Button.UP, Button.DOWN, Button.LEFT, Button.RIGHT})
FACE: Final = frozenset({Button.A, Button.B})
SHOULDER: Final = frozenset({Button.L, Button.R})
SYSTEM: Final = frozenset({Button.START, Button.SELECT})

OPPOSED_PAIRS: Final[tuple[tuple[Button, Button], ...]] = (
    (Button.UP, Button.DOWN),
    (Button.LEFT, Button.RIGHT),
)
"""D-pad pairs that cannot be held together on a physical pad.

Only the two D-pad axes. L+R is a perfectly ordinary simultaneous press -- several games
use it as a deliberate chord -- so the shoulders are deliberately absent here.

Real hardware *does* let both bits of an axis go low at once (the pad is four independent
switches), and some games rely on it while others glitch. Policy lives in `safety`.
"""

# Keys that look like GBA buttons to someone reading an emulator's config screen, or that
# belong to a different console. Named explicitly so the error can say why.
_NOT_GBA_INPUTS: Final[dict[str, str]] = {
    "x": "the GBA has no X button (that is SNES); its face buttons are 'a' and 'b'",
    "y": "the GBA has no Y button (that is SNES); its face buttons are 'a' and 'b'",
    "space": "spacebar is not a GBA input; emulators usually bind Start to Enter",
    "spacebar": "spacebar is not a GBA input; emulators usually bind Start to Enter",
    "lb": "use 'l' for the left shoulder button",
    "rb": "use 'r' for the right shoulder button",
    "l1": "use 'l' for the left shoulder button",
    "r1": "use 'r' for the right shoulder button",
    "l2": "the GBA has only one shoulder button per side: 'l' and 'r'",
    "r2": "the GBA has only one shoulder button per side: 'l' and 'r'",
    "z": "'z' is a keyboard key, not a GBA input; in the 'wasd' dialect it means 'b'",
}

_CANON: Final[dict[str, Button]] = {b.value: b for b in Button}

# mGBA desktop defaults: arrows for the D-pad, X=A, Z=B, A=L, S=R, Enter=Start,
# Backspace=Select. Note that here "a" is the *shoulder*, not a direction or a face button.
_MGBA: Final[dict[str, Button]] = {
    "up": Button.UP,
    "down": Button.DOWN,
    "left": Button.LEFT,
    "right": Button.RIGHT,
    "x": Button.A,
    "z": Button.B,
    "a": Button.L,
    "s": Button.R,
    "enter": Button.START,
    "return": Button.START,
    "backspace": Button.SELECT,
}

# WASD + ZXCV, the layout most people mean by "the keyboard controls". On the GBA this
# covers eight of the ten inputs; only Start and Select fall outside it.
_WASD: Final[dict[str, Button]] = {
    "w": Button.UP,
    "s": Button.DOWN,
    "a": Button.LEFT,
    "d": Button.RIGHT,
    "x": Button.A,
    "z": Button.B,
    "c": Button.L,
    "v": Button.R,
    "enter": Button.START,
    "return": Button.START,
    "backspace": Button.SELECT,
}

DIALECTS: Final[dict[str, dict[str, Button]]] = {
    "canon": _CANON,
    "mgba": _MGBA,
    "wasd": _WASD,
}
"""Token vocabularies. ``canon`` is the only one the agent-facing API ever uses."""


class UnknownButtonError(ValueError):
    """A token could not be resolved to one of the ten GBA inputs.

    The message lists every valid name on purpose: when an agent or a human hits this, the
    error text should be sufficient to recover without going and reading documentation.
    """

    def __init__(self, token: str, dialect: str = "canon") -> None:
        self.token = token
        self.dialect = dialect
        valid = ", ".join(b.value for b in CANON_ORDER)
        hint = _NOT_GBA_INPUTS.get(token.strip().lower())
        parts = [f"unknown button {token!r}"]
        if dialect != "canon":
            parts.append(f"in dialect {dialect!r}")
        parts.append(f"-- the Game Boy Advance has exactly 10 inputs: {valid}")
        if hint:
            parts.append(f"({hint})")
        super().__init__(" ".join(parts))


def parse(token: str | Button, *, dialect: str = "canon") -> Button:
    """Resolve one token to a `Button`.

    Case and surrounding whitespace are ignored. `Button` instances pass through, so this
    is safe to call on already-parsed input.

    Args:
        token: a canonical name (``"a"``, ``"up"``, ``"l"``, ...) or, with an explicit
            non-canon `dialect`, a keyboard key name. Remember that canonical ``"l"`` and
            ``"r"`` are the shoulder buttons; the directions are ``"left"`` and
            ``"right"``.
        dialect: a key of `DIALECTS`. Leave as ``"canon"`` unless you are deliberately
            accepting keyboard input from a human.

    Raises:
        UnknownButtonError: if the token is not a button in that dialect.
        ValueError: if `dialect` is not a known dialect.
    """
    if isinstance(token, Button):
        return token
    try:
        table = DIALECTS[dialect]
    except KeyError:
        known = ", ".join(sorted(DIALECTS))
        raise ValueError(f"unknown dialect {dialect!r}; expected one of: {known}") from None
    key = token.strip().lower()
    try:
        return table[key]
    except KeyError:
        raise UnknownButtonError(token, dialect) from None


def parse_set(tokens: object, *, dialect: str = "canon") -> frozenset[Button]:
    """Resolve an iterable of tokens (or a single token) to a frozen set of buttons.

    A bare string is treated as one token rather than as an iterable of characters --
    ``parse_set("start")`` is the Start button, not five unknown buttons.

    Raises:
        UnknownButtonError: on the first token that does not resolve.
        TypeError: if `tokens` is not a string, `Button`, or iterable of them.
    """
    if isinstance(tokens, str | Button):
        return frozenset({parse(tokens, dialect=dialect)})
    try:
        items = list(tokens)  # type: ignore[call-overload]
    except TypeError:
        raise TypeError(
            f"expected a button name or an iterable of button names, got {type(tokens).__name__}"
        ) from None
    return frozenset(parse(t, dialect=dialect) for t in items)


def canon_sort(buttons: frozenset[Button] | set[Button] | list[Button]) -> tuple[Button, ...]:
    """Order a button set canonically.

    Hardware does not care about ordering -- the core reads the whole held set once per
    frame -- but determinism does, for transcripts, replay and test assertions. Everything
    that emits a sequence of edges uses this.
    """
    return tuple(sorted(buttons, key=_CANON_INDEX.__getitem__))
