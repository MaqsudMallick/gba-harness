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

case "$(uname -s)" in
  Darwin)
    case "$(uname -m)" in
      arm64) platform="arm64" ;;
      x86_64) platform="x86_64" ;;
      *) echo "error: unsupported macOS architecture: $(uname -m)" >&2; exit 1 ;;
    esac
    archive="$work/mgba_libretro.dylib.zip"
    url="https://buildbot.libretro.com/nightly/apple/osx/$platform/latest/mgba_libretro.dylib.zip"
    echo "downloading the official Libretro macOS $platform mGBA core..."
    curl --fail --location --silent --show-error "$url" --output "$archive"
    mkdir -p "$work/x" "$dest"
    unzip -q "$archive" -d "$work/x"
    core="$(find "$work/x" -name 'mgba_libretro.dylib' -print -quit)"
    [ -n "$core" ] || { echo "error: mgba_libretro.dylib not found in the archive" >&2; exit 1; }
    install -m 0644 "$core" "$dest/mgba_libretro.dylib"
    ;;
  Linux)
    command -v apt >/dev/null || {
      echo "error: automatic Linux fetching currently requires apt" >&2
      echo "set GBHARNESS_LIBRETRO_CORE to an existing mGBA libretro core instead" >&2
      exit 1
    }
    echo "downloading libretro-mgba..."
    ( cd "$work" && apt download libretro-mgba >/dev/null )
    mkdir -p "$work/x" "$dest"
    dpkg-deb -x "$work"/libretro-mgba_*.deb "$work/x"
    core="$(find "$work/x" -name 'mgba_libretro.so' -print -quit)"
    [ -n "$core" ] || { echo "error: mgba_libretro.so not found in the package" >&2; exit 1; }
    install -m 0644 "$core" "$dest/mgba_libretro.so"
    ;;
  *)
    echo "error: unsupported operating system: $(uname -s)" >&2
    echo "set GBHARNESS_LIBRETRO_CORE to an existing mGBA libretro core instead" >&2
    exit 1
    ;;
esac

echo "core installed at: $dest/$(basename "$core")"
echo "the test suite finds it automatically; for your own scripts, export:"
echo "  export GBHARNESS_LIBRETRO_CORE=$dest/$(basename "$core")"
