#!/bin/bash
# Drag a folder of green-screen frames after `ck` — see scripts/ck.py.
export CK_CALLER_CWD="$PWD"
cd "$(dirname "$(readlink -f "$0" 2>/dev/null || realpath "$0")")" || exit 1
exec uv run --extra mlx python scripts/ck.py "$@"
