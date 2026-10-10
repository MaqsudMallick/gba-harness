"""A tiny Game Boy Advance ROM that paints the screen with what it reads from KEYINPUT.

Hand-assembled ARM machine code -- there is no cross-assembler on this machine, and for
fourteen instructions a table of encodings is clearer than a toolchain dependency anyway.

Why this exists: libretro does not expose the GBA's KEYINPUT register for read-back (see
PLAN-input-layer.md Appendix A), so unlike the Game Boy version of this harness there is no
register to assert against. A cartridge that *reports what it saw* restores that oracle from
the other side -- and makes the input chain visible, which a spinning test ROM cannot.

The program writes one **indicator pixel per button**: pixel *i* is white if button *i* is
held and black if it is not.

That spatial encoding is deliberate. The obvious version -- paint the whole screen with the
button bitmask as a colour -- was tried first and *loses information*: the core converts
VRAM's BGR555 to the frontend's RGB565, and repacking 5-bit channels into 5/6/5 silently
drops bits. Four of the ten buttons became indistinguishable from "nothing pressed". Black
and white survive any pixel format, so the oracle is exact.

The program, in assembly::

            mov  r0, #0x04000000    @ IO base
            add  r4, r0, #0x130     @ &KEYINPUT
            mov  r1, #0x400
            orr  r1, r1, #3
            str  r1, [r0]           @ DISPCNT = mode 3 (240x160x16bpp bitmap) + BG2 on
            mov  r2, #0x06000000    @ VRAM
    loop:   ldrh r3, [r4]           @ KEYINPUT: a 0 bit means pressed
            mvn  r3, r3             @ invert, so a 1 bit means pressed
            mov  r5, r2
            mov  r7, r3
            mov  r6, #10            @ ten buttons
    bit:    and  r8, r7, #1
            rsb  r8, r8, #0         @ 0 -> 0x0000 (black), 1 -> 0xFFFF (white)
            strh r8, [r5], #2
            mov  r7, r7, lsr #1
            subs r6, r6, #1
            bne  bit
            b    loop

Pixel order follows the hardware KEYINPUT bit order, which is *not* this package's canonical
order -- see `KEYINPUT_BITS`.
"""

from __future__ import annotations

from typing import Final

ROM_SIZE: Final = 32 * 1024
CODE_ORIGIN: Final = 0xC0  # first byte after the 192-byte cartridge header

#: GBA hardware KEYINPUT bit order (GBATEK). Deliberately not this package's canonical
#: order -- the ROM reports in hardware order and the demo maps back.
KEYINPUT_BITS: Final[tuple[str, ...]] = (
    "a", "b", "select", "start", "right", "left", "up", "down", "r", "l",
)

#: (encoding, mnemonic) in execution order, starting at CODE_ORIGIN.
PROGRAM: Final[tuple[tuple[int, str], ...]] = (
    (0xE3A00404, "mov  r0, #0x04000000"),
    (0xE2804E13, "add  r4, r0, #0x130"),
    (0xE3A01B01, "mov  r1, #0x400"),
    (0xE3811003, "orr  r1, r1, #3"),
    (0xE5801000, "str  r1, [r0]"),
    (0xE3A02406, "mov  r2, #0x06000000"),
    (0xE1D430B0, "loop: ldrh r3, [r4]"),
    (0xE1E03003, "mvn  r3, r3"),
    (0xE1A05002, "mov  r5, r2"),
    (0xE1A07003, "mov  r7, r3"),
    (0xE3A0600A, "mov  r6, #10"),
    (0xE2078001, "bit:  and  r8, r7, #1"),
    (0xE2688000, "rsb  r8, r8, #0"),
    (0xE0C580B2, "strh r8, [r5], #2"),
    (0xE1A070A7, "mov  r7, r7, lsr #1"),
    (0xE2566001, "subs r6, r6, #1"),
    (0x1AFFFFF9, "bne  bit"),
    (0xEAFFFFF3, "b    loop"),
)

#: `b CODE_ORIGIN` from the cartridge entry point at offset 0.
#: offset = (target - (pc + 8)) / 4 = (0xC0 - 0x08) / 4 = 46
_ENTRY_BRANCH: Final = 0xEA000000 | ((CODE_ORIGIN - 8) // 4)


def build_rom(size: int = ROM_SIZE) -> bytes:
    """Assemble the ROM image.

    The 156-byte Nintendo logo at 0x04 is left zeroed: real hardware's BIOS verifies it,
    but an emulator running the HLE BIOS does not, so nothing copyrighted is needed.
    """
    rom = bytearray(size)
    rom[0x00:0x04] = _ENTRY_BRANCH.to_bytes(4, "little")
    rom[0xA0:0xAC] = b"KEYINPUT   "[:12].ljust(12, b"\x00")
    rom[0xAC:0xB0] = b"KEYI"
    rom[0xB0:0xB2] = b"00"
    rom[0xB2] = 0x96  # fixed value, checked by real BIOS

    for index, (encoding, _) in enumerate(PROGRAM):
        at = CODE_ORIGIN + index * 4
        rom[at : at + 4] = encoding.to_bytes(4, "little")

    checksum = 0
    for addr in range(0xA0, 0xBD):
        checksum = (checksum - rom[addr]) & 0xFF
    rom[0xBD] = (checksum - 0x19) & 0xFF
    return bytes(rom)


def disassemble() -> str:
    """The program with addresses, for the demo to print."""
    return "\n".join(
        f"  0x{0x08000000 + CODE_ORIGIN + i * 4:08X}  {enc:08X}  {text}"
        for i, (enc, text) in enumerate(PROGRAM)
    )


if __name__ == "__main__":
    import pathlib
    import sys

    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "roms/keyinput.gba")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(build_rom())
    print(f"wrote {out} ({ROM_SIZE} bytes)")
    print(disassemble())
