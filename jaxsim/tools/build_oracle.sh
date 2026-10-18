#!/usr/bin/env bash
# Build the C++ parity oracle into jaxsim/oracle/build/jaxsim_oracle.
# macOS: set SDKROOT if the default Command Line Tools SDK does not link (e.g. SDKROOT=.../MacOSX26.5.sdk).
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
args=(-S "$here/oracle" -B "$here/oracle/build" -DCMAKE_BUILD_TYPE=Release)
command -v ninja >/dev/null && args+=(-G Ninja)
[[ -n "${SDKROOT:-}" ]] && args+=(-DCMAKE_OSX_SYSROOT="$SDKROOT")
cmake "${args[@]}" >/dev/null
cmake --build "$here/oracle/build" --target jaxsim_oracle
