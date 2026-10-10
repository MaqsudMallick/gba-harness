"""The assembler is checked against encodings that are known to execute.

`examples/keyinput_rom.py` was hand-encoded and independently verified on the real core --
all ten buttons round-trip through it. Reproducing every one of its instructions is a genuine
oracle for the assembler, not a restatement of the same assumptions.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from typing import Any

import pytest

_EXAMPLES = pathlib.Path(__file__).parent.parent / "examples"


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, _EXAMPLES / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def a() -> Any:
    return _load("gba_asm")


@pytest.fixture(scope="module")
def rom() -> Any:
    return _load("keyinput_rom")


# --- immediate encoding -------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0x04000000, 0x301),  # see test_multiple_encodings_are_valid
        (0x06000000, 0x406),
        (0x400, 0xB01),
        (0x130, 0xE13),
        (0x9600, 0xC96),
        (0, 0x000),
        (1, 0x001),
        (0xFF, 0x0FF),
    ],
)
def test_immediate_encoding(a, value, expected):
    assert a.imm12(value) == expected


def test_unencodable_immediate_is_a_clear_error(a):
    # 0x3FF needs 10 contiguous bits; ARM immediates are 8 bits rotated by an even amount.
    with pytest.raises(ValueError, match="not encodable"):
        a.imm12(0x3FF)


def test_immediate_round_trips(a):
    for value in (0, 1, 2, 4, 0xFF, 0x100, 0x400, 0x130, 0x9600, 0x04000000, 0x0E000000):
        encoded = a.imm12(value)
        rot, imm8 = encoded >> 8, encoded & 0xFF
        n = (2 * rot) & 31
        rotated = imm8 if n == 0 else ((imm8 >> n) | (imm8 << (32 - n))) & 0xFFFFFFFF
        assert rotated == value


def test_multiple_encodings_are_valid_and_we_pick_the_canonical_one(a):
    """`0x04000000` is encodable two ways, and only one is canonical.

    `imm8=1, rot=3` and `imm8=4, rot=4` both denote bit 26. The ARM ARM requires the
    smallest rotation, so the assembler emits the first; the hand-written constant in
    `keyinput_rom.py` happens to use the second. Both execute identically, which is why
    the comparison below is by meaning rather than by bytes.
    """
    assert a.imm12(0x04000000) == 0x301
    assert _decode_imm(0x404) == _decode_imm(0x301) == 0x04000000


def _decode_imm(imm12_field: int) -> int:
    rot, imm8 = (imm12_field >> 8) & 0xF, imm12_field & 0xFF
    n = (2 * rot) & 31
    return imm8 if n == 0 else ((imm8 >> n) | (imm8 << (32 - n))) & 0xFFFFFFFF


def _equivalent(got: int, want: int) -> bool:
    """Equal, or differing only in which valid encoding of the same immediate was chosen."""
    if got == want:
        return True
    if (got >> 12) != (want >> 12):  # cond, I, opcode, S, Rn, Rd must all match
        return False
    is_immediate = (got >> 25) & 1
    return bool(is_immediate) and _decode_imm(got & 0xFFF) == _decode_imm(want & 0xFFF)


# --- the oracle: rebuild a ROM known to run -----------------------------------------


def test_assembler_reproduces_every_keyinput_rom_instruction(a, rom):
    rebuilt = [
        a.dp("mov", 0, imm=0x04000000),
        a.dp("add", 4, 0, imm=0x130),
        a.dp("mov", 1, imm=0x400),
        a.dp("orr", 1, 1, imm=3),
        a.ldr_str(False, 1, 0, 0),
        a.dp("mov", 2, imm=0x06000000),
        a.ldrh_strh(True, 3, 4, 0),
        a.dp("mvn", 3, rm=3),
        a.dp("mov", 5, rm=2),
        a.dp("mov", 7, rm=3),
        a.dp("mov", 6, imm=10),
        a.dp("and", 8, 7, imm=1),
        a.dp("rsb", 8, 8, imm=0),
        a.ldrh_strh(False, 8, 5, 2, post=True),
        a.dp("mov", 7, rm=7, shift="lsr", amount=1),
        a.dp("sub", 6, 6, imm=1, s=True),
        a.branch(-20, cond="ne"),
        a.branch(-44),
    ]
    expected = [encoding for encoding, _ in rom.PROGRAM]
    assert len(rebuilt) == len(expected)
    for index, (got, want) in enumerate(zip(rebuilt, expected, strict=True)):
        assert _equivalent(got, want), (
            f"instruction {index} ({rom.PROGRAM[index][1]}): "
            f"assembled {got:08X}, hand-encoded {want:08X}"
        )


def test_entry_branch_matches(a, rom):
    # `b 0xC0` from offset 0: the ROM's own entry point.
    assert a.branch(rom.CODE_ORIGIN) == 0xEA00002E


# --- comparison instructions force S and ignore Rd ----------------------------------


def test_tst_sets_the_s_bit(a):
    word = a.dp("tst", rn=3, imm=0x10)
    assert word >> 20 & 1, "tst must set flags or the following condition is meaningless"


def test_conditional_execution_encodes_the_condition(a):
    assert a.dp("add", 5, 5, imm=4, cond="ne") >> 28 == 1
    assert a.dp("mov", 5, imm=0, cond="lt") >> 28 == 11


# --- transfers ----------------------------------------------------------------------


def test_halfword_offset_is_split_across_two_nibbles(a):
    word = a.ldrh_strh(False, 8, 5, 0x2F, post=True)
    assert (word >> 8 & 0xF) == 0x2, "high nibble"
    assert (word & 0xF) == 0xF, "low nibble"


def test_oversized_halfword_offset_is_rejected(a):
    with pytest.raises(ValueError, match="out of range"):
        a.ldrh_strh(False, 0, 0, 0x100)


def test_mul_rejects_the_forbidden_register_pair(a):
    # ARM7 gives unpredictable results when Rd == Rm; better to refuse than to emit it.
    with pytest.raises(ValueError, match="different registers"):
        a.mul(8, 8, 7)


# --- labels -------------------------------------------------------------------------


def test_backward_label_branch_matches_a_manual_offset(a):
    asm = a.Assembler(origin=0x080000C0)
    asm.label("top")
    asm.emit(a.dp("mov", 0, imm=1), "mov r0, #1")
    asm.branch_to("top")
    words = asm.assemble()
    assert words[1] == a.branch(-4)


def test_forward_label_branch_resolves(a):
    asm = a.Assembler(origin=0)
    asm.branch_to("end")
    asm.emit(a.dp("mov", 0, imm=1), "mov r0, #1")
    asm.label("end")
    asm.emit(a.dp("mov", 1, imm=2), "mov r1, #2")
    assert asm.assemble()[0] == a.branch(8)


def test_undefined_label_is_an_error(a):
    asm = a.Assembler()
    asm.branch_to("nowhere")
    with pytest.raises(ValueError, match="undefined label"):
        asm.assemble()


def test_duplicate_label_is_an_error(a):
    asm = a.Assembler()
    asm.label("x")
    with pytest.raises(ValueError, match="duplicate label"):
        asm.label("x")
