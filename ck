#!/bin/bash
# Drag a folder of green-screen frames after `ck` — see scripts/ck.py.
export CK_CALLER_CWD="$PWD"
cd "$(dirname "$(readlink -f "$0" 2>/dev/null || realpath "$0")")" || exit 1
EXTRA=()
# Apple Vision bindings for automatic alpha hints (--auto-hint), macOS only.
[ "$(uname)" = "Darwin" ] && EXTRA=(--with pyobjc-framework-Vision --with pyobjc-framework-Quartz)
exec uv run --extra mlx "${EXTRA[@]}" python scripts/ck.py "$@"
