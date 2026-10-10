"""Milestone 2: the ten inputs, aliases, and the error that teaches."""

from __future__ import annotations

import pytest

from gbharness.buttons import (
    ALL,
    CANON_ORDER,
    DPAD,
    FACE,
    OPPOSED_PAIRS,
    SHOULDER,
    SYSTEM,
    Button,
    UnknownButtonError,
    canon_sort,
    parse,
    parse_set,
)


def test_there_are_exactly_ten_inputs():
    assert len(Button) == 10
    assert len(ALL) == 10
    assert DPAD | FACE | SHOULDER | SYSTEM == ALL
    groups = [DPAD, FACE, SHOULDER, SYSTEM]
    for i, a in enumerate(groups):
        for b in groups[i + 1 :]:
            assert not (a & b), "button groups must not overlap"


def test_group_sizes():
    assert len(DPAD) == 4
    assert len(FACE) == 2 and FACE == {Button.A, Button.B}
    assert len(SHOULDER) == 2 and SHOULDER == {Button.L, Button.R}
    assert len(SYSTEM) == 2


@pytest.mark.parametrize(
    "name", ["up", "down", "left", "right", "a", "b", "l", "r", "start", "select"]
)
def test_every_canonical_name_parses(name):
    assert parse(name).value == name


def test_there_is_no_x_or_y_on_a_gba():
    for token in ("x", "y"):
        with pytest.raises(UnknownButtonError, match="SNES"):
            parse(token)


# --- the l/left trap ----------------------------------------------------------------


def test_l_and_r_are_shoulders_not_directions():
    assert parse("l") is Button.L
    assert parse("r") is Button.R
    assert parse("left") is Button.LEFT
    assert parse("right") is Button.RIGHT
    assert parse("l") is not Button.LEFT
    assert parse("r") is not Button.RIGHT


def test_shoulders_are_not_an_opposed_pair():
    # L+R is an ordinary chord that several games use deliberately. Only the D-pad axes
    # are physically exclusive.
    for pair in OPPOSED_PAIRS:
        assert not (set(pair) & SHOULDER)
    assert OPPOSED_PAIRS == ((Button.UP, Button.DOWN), (Button.LEFT, Button.RIGHT))


@pytest.mark.parametrize(
    ("token", "fragment"), [("lb", "'l'"), ("rb", "'r'"), ("l1", "'l'"), ("r1", "'r'")]
)
def test_gamepad_shoulder_names_are_redirected(token, fragment):
    with pytest.raises(UnknownButtonError, match="shoulder"):
        parse(token)


def test_there_is_only_one_shoulder_per_side():
    for token in ("l2", "r2"):
        with pytest.raises(UnknownButtonError, match="only one shoulder"):
            parse(token)


# --- dialects -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "expected"),
    [("A", Button.A), ("  Start  ", Button.START), ("SELECT", Button.SELECT), ("L", Button.L)],
)
def test_case_and_whitespace_are_ignored(token, expected):
    assert parse(token) is expected


def test_button_instances_pass_through():
    assert parse(Button.B) is Button.B


def test_the_letter_a_means_three_different_things_across_dialects():
    """The collision that makes dialects opt-in. Each reading is correct in its own."""
    assert parse("a") is Button.A  # the A face button
    assert parse("a", dialect="wasd") is Button.LEFT  # the D-pad direction
    assert parse("a", dialect="mgba") is Button.L  # the left *shoulder*


def test_wasd_dialect_covers_eight_of_the_ten_inputs():
    # WASD + ZXCV really is most of a GBA pad -- only Start and Select fall outside it.
    mapped = {parse(k, dialect="wasd") for k in ("w", "a", "s", "d", "z", "x", "c", "v")}
    assert mapped == ALL - SYSTEM
    assert parse("c", dialect="wasd") is Button.L
    assert parse("v", dialect="wasd") is Button.R


def test_mgba_dialect_matches_its_desktop_defaults():
    assert parse("x", dialect="mgba") is Button.A
    assert parse("z", dialect="mgba") is Button.B
    assert parse("a", dialect="mgba") is Button.L
    assert parse("s", dialect="mgba") is Button.R
    assert parse("enter", dialect="mgba") is Button.START
    assert parse("backspace", dialect="mgba") is Button.SELECT


def test_keyboard_keys_are_rejected_in_canon_dialect():
    for token in ("w", "c", "v", "enter", "backspace"):
        with pytest.raises(UnknownButtonError):
            parse(token)


def test_unknown_dialect_is_a_value_error_not_a_button_error():
    with pytest.raises(ValueError, match="unknown dialect"):
        parse("a", dialect="dvorak")


# --- the error that teaches ---------------------------------------------------------


def test_error_message_lists_all_ten_valid_names():
    with pytest.raises(UnknownButtonError) as exc:
        parse("q")
    message = str(exc.value)
    assert "10 inputs" in message
    for name in (b.value for b in Button):
        assert name in message, f"{name!r} missing from error text"


@pytest.mark.parametrize(
    ("token", "fragment"),
    [
        ("x", "SNES"),
        ("y", "SNES"),
        ("space", "Enter"),
        ("spacebar", "Enter"),
        ("z", "wasd"),
    ],
)
def test_near_miss_keys_explain_themselves(token, fragment):
    with pytest.raises(UnknownButtonError) as exc:
        parse(token)
    assert fragment in str(exc.value)


# --- parse_set ----------------------------------------------------------------------


def test_parse_set_treats_a_bare_string_as_one_token():
    assert parse_set("start") == frozenset({Button.START})
    assert parse_set("l") == frozenset({Button.L})


def test_parse_set_from_iterable():
    assert parse_set(["up", "a", "r"]) == frozenset({Button.UP, Button.A, Button.R})
    assert parse_set(()) == frozenset()


def test_parse_set_deduplicates():
    assert parse_set(["a", "A", Button.A]) == frozenset({Button.A})


def test_parse_set_rejects_non_iterable():
    with pytest.raises(TypeError):
        parse_set(42)


def test_parse_set_reports_the_bad_token():
    with pytest.raises(UnknownButtonError) as exc:
        parse_set(["up", "nope"])
    assert "nope" in str(exc.value)


# --- ordering -----------------------------------------------------------------------


def test_canon_sort_is_declaration_order_and_stable():
    assert canon_sort({Button.SELECT, Button.UP, Button.A, Button.R}) == (
        Button.UP,
        Button.A,
        Button.R,
        Button.SELECT,
    )
    assert canon_sort(ALL) == CANON_ORDER
    assert canon_sort({Button.A, Button.UP}) == canon_sort({Button.UP, Button.A})


def test_canonical_order_groups_dpad_face_shoulder_system():
    names = [b.value for b in CANON_ORDER]
    assert names == ["up", "down", "left", "right", "a", "b", "l", "r", "start", "select"]


def test_button_is_a_str_enum_so_it_serializes_as_its_name():
    assert Button.A == "a"
    assert Button.L == "l"
    assert f"{Button.START}" == "start"
