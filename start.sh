#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")" && pwd)
exec python3 "$root/launch.py" start "$@"
