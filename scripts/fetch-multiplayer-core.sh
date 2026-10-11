#!/usr/bin/env bash
# Build the mGBA Splitscreen libretro core into vendor/libretro without root.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/vendor/libretro"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

command -v git >/dev/null || { echo "error: git is required" >&2; exit 1; }
command -v cmake >/dev/null || {
  echo "error: cmake is required (macOS: brew install cmake)" >&2
  exit 1
}

repo="https://github.com/Spuds0588/mgba-splitscreen.git"
ref="${GBHARNESS_MGBA_SPLITSCREEN_REF:-b66511a35873bda9c961872813e58b35543bcfc5}"
echo "fetching mGBA Splitscreen at $ref..."
git clone --filter=blob:none --no-checkout "$repo" "$work/src"
git -C "$work/src" fetch --depth 1 origin "$ref"
git -C "$work/src" checkout --detach FETCH_HEAD

cmake -S "$work/src" -B "$work/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_FLAGS_RELEASE='-O2 -DNDEBUG -DCOLOR_16_BIT -DCOLOR_5_6_5' \
  -DLIBMGBA_ONLY=ON \
  -DBUILD_LIBRETRO_SPLITSCREEN=ON
cmake --build "$work/build" --target mgba_splitscreen_libretro -j 8

case "$(uname -s)" in
  Darwin) artifact="$work/build/mgba_splitscreen_libretro.dylib" ;;
  Linux) artifact="$work/build/mgba_splitscreen_libretro.so" ;;
  *) echo "error: unsupported operating system: $(uname -s)" >&2; exit 1 ;;
esac

mkdir -p "$dest"
install -m 0755 "$artifact" "$dest/$(basename "$artifact")"
echo "multiplayer core installed at: $dest/$(basename "$artifact")"
