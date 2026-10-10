"""A very small ARM assembler — just enough to build the demo cartridges.

There is no ARM cross-assembler on this machine, and the alternative is typing 32-bit hex
constants by hand. For the 18-instruction KEYINPUT reporter that was defensible; for a
45-instruction game it is not, so the encodings are generated and checked instead.

Covers only the subset the demo ROMs use: data processing with an immediate or a register,
halfword and word transfers, multiply, and branches. It is validated against known-good
encodings in `tests/test_gba_asm.py`, including every instruction of the KEYINPUT ROM,
which was hand-encoded and independently verified to run.
"""

from __future__ import annotations

from typing import Final

COND: Final[dict[str, int]] = {
    "eq": 0, "ne": 1, "cs": 2, "cc": 3, "mi": 4, "pl": 5, "vs": 6, "vc": 7,
    "hi": 8, "ls": 9, "ge": 10, "lt": 11, "gt": 12, "le": 13, "al": 14,
}
_OPCODE: Final[dict[str, int]] = {
    "and": 0, "eor": 1, "sub": 2, "rsb": 3, "add": 4, "adc": 5, "sbc": 6, "rsc": 7,
    "tst": 8, "teq": 9, "cmp": 10, "cmn": 11, "orr": 12, "mov": 13, "bic": 14, "mvn": 15,
}
#: These compare rather than write a result: S is forced on and Rd is unused.
_NO_DEST: Final = frozenset({"tst", "teq", "cmp", "cmn"})
#: These take no source register: Rn is unused.
_NO_SRC: Final = frozenset({"mov", "mvn"})

SHIFT: Final[dict[str, int]] = {"lsl": 0, "lsr": 1, "asr": 2, "ror": 3}


def _rol32(value: int, n: int) -> int:
    n &= 31
    if n == 0:
        return value & 0xFFFFFFFF
    return ((value << n) | (value >> (32 - n))) & 0xFFFFFFFF


def imm12(value: int) -> int:
    """Encode an ARM data-processing immediate: an 8-bit value rotated right by an even amount.

    Raises:
        ValueError: if the value is not representable, which is a real and common
            constraint -- 0x3FF, for instance, cannot be encoded and needs two instructions.
    """
    if not 0 <= value <= 0xFFFFFFFF:
        raise ValueError(f"immediate {value:#x} out of range")
    for rot in range(16):
        candidate = _rol32(value, 2 * rot)
        if candidate <= 0xFF:
            return (rot << 8) | candidate
    raise ValueError(
        f"{value:#x} is not encodable as an ARM immediate "
        "(8-bit value rotated right by an even amount); build it in two instructions"
    )


def dp(op: str, rd: int = 0, rn: int = 0, *, imm: int | None = None, rm: int | None = None,
       shift: str = "lsl", amount: int = 0, s: bool = False, cond: str = "al") -> int:
    """Data processing: `mov`, `add`, `sub`, `orr`, `tst`, `cmp`, `mvn`, ..."""
    opcode = _OPCODE[op]
    set_flags = s or op in _NO_DEST
    if imm is not None:
        i_bit, operand2 = 1, imm12(imm)
    elif rm is not None:
        i_bit = 0
        operand2 = (amount & 0x1F) << 7 | SHIFT[shift] << 5 | (rm & 0xF)
    else:
        raise ValueError("dp needs either imm= or rm=")
    return (
        COND[cond] << 28 | i_bit << 25 | opcode << 21 | int(set_flags) << 20
        | (rn & 0xF) << 16 | (rd & 0xF) << 12 | operand2
    )


def ldr_str(load: bool, rd: int, rn: int, offset: int = 0, *, post: bool = False,
            cond: str = "al") -> int:
    """Word transfer with an immediate offset."""
    up = offset >= 0
    off = abs(offset)
    if off > 0xFFF:
        raise ValueError(f"word offset {offset} out of range")
    p_bit = 0 if post else 1
    return (
        COND[cond] << 28 | 0b01 << 26 | p_bit << 24 | int(up) << 23 | int(load) << 20
        | (rn & 0xF) << 16 | (rd & 0xF) << 12 | off
    )


def ldrh_strh(load: bool, rd: int, rn: int, offset: int = 0, *, post: bool = False,
              cond: str = "al") -> int:
    """Halfword transfer with an immediate offset. The offset is split across two nibbles."""
    up = offset >= 0
    off = abs(offset)
    if off > 0xFF:
        raise ValueError(f"halfword offset {offset} out of range (8-bit split field)")
    p_bit = 0 if post else 1
    return (
        COND[cond] << 28 | p_bit << 24 | int(up) << 23 | 1 << 22 | int(load) << 20
        | (rn & 0xF) << 16 | (rd & 0xF) << 12
        | (off >> 4) << 8 | 0b1011 << 4 | (off & 0xF)
    )


def mul(rd: int, rm: int, rs: int, *, cond: str = "al") -> int:
    """`mul rd, rm, rs`. On ARM7, rd and rm must differ."""
    if rd == rm:
        raise ValueError("mul: destination and first operand must be different registers")
    return COND[cond] << 28 | (rd & 0xF) << 16 | (rs & 0xF) << 8 | 0b1001 << 4 | (rm & 0xF)


def branch(offset_bytes: int, *, link: bool = False, cond: str = "al") -> int:
    """Branch by a byte offset relative to the instruction's own address.

    The ARM pipeline means the encoded offset is relative to `pc + 8`; that adjustment is
    applied here so callers work in plain "target minus here" byte distances.
    """
    words = (offset_bytes - 8) >> 2
    return COND[cond] << 28 | 0b101 << 25 | int(link) << 24 | (words & 0xFFFFFF)


class Assembler:
    """Collects instructions and resolves labels in a second pass."""

    def __init__(self, origin: int = 0) -> None:
        self.origin = origin
        self._words: list[int | tuple[str, str, bool]] = []
        self._labels: dict[str, int] = {}
        self._text: list[str] = []

    @property
    def here(self) -> int:
        return self.origin + 4 * len(self._words)

    def label(self, name: str) -> None:
        if name in self._labels:
            raise ValueError(f"duplicate label {name!r}")
        self._labels[name] = self.here

    def emit(self, word: int, text: str) -> None:
        self._words.append(word)
        self._text.append(text)

    def branch_to(self, name: str, *, link: bool = False, cond: str = "al") -> None:
        """A branch to a label that may not exist yet."""
        self._words.append((name, cond, link))
        self._text.append(f"b{'l' if link else ''}{'' if cond == 'al' else cond} {name}")

    def assemble(self) -> list[int]:
        out: list[int] = []
        for index, word in enumerate(self._words):
            if isinstance(word, int):
                out.append(word)
                continue
            name, cond, link = word
            if name not in self._labels:
                raise ValueError(f"undefined label {name!r}")
            here = self.origin + 4 * index
            out.append(branch(self._labels[name] - here, link=link, cond=cond))
        return out

    def listing(self) -> str:
        words = self.assemble()
        by_addr = {addr: name for name, addr in self._labels.items()}
        lines = []
        for index, (word, text) in enumerate(zip(words, self._text, strict=True)):
            addr = self.origin + 4 * index
            tag = f"{by_addr[addr]}:" if addr in by_addr else ""
            lines.append(f"  {addr:08X}  {word:08X}  {tag:<8}{text}")
        return "\n".join(lines)
