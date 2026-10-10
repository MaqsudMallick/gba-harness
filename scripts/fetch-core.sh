#!/usr/bin/env bash
# Fetch the mGBA libretro core into vendor/ without installing it system-wide.
#
# The emulator is a system artifact, not a Python dependency, so it is not in
# pyproject.toml. Either install it globally:
#
#     sudo apt install libretro-mgba
#
# ...or run this script, which downloads the same .deb and extracts just the core.
# No root required. The result is gitignored.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/vendor/libretro"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "downloading libretro-mgba..."
( cd "$work" && apt download libretro-mgba >/dev/null )

mkdir -p "$work/x" "$dest"
dpkg-deb -x "$work"/libretro-mgba_*.deb "$work/x"

core="$(find "$work/x" -name 'mgba_libretro.so' -print -quit)"
[ -n "$core" ] || { echo "error: mgba_libretro.so not found in the package" >&2; exit 1; }
install -m 0644 "$core" "$dest/mgba_libretro.so"

echo "core installed at: $dest/mgba_libretro.so"
echo "the test suite finds it automatically; for your own scripts, export:"
echo "  export GBHARNESS_LIBRETRO_CORE=$dest/mgba_libretro.so"
