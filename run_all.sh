#!/usr/bin/env bash
# One-command reproduction. Usage:  bash run_all.sh [--quick]
#
# STATUS: partial (Milestone 1). It currently only records the environment, validates the
# datasets and runs the unit tests. Benchmark stages are added in later milestones.
set -euo pipefail
cd "$(dirname "$0")"

QUICK=0
for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 64 ;;
  esac
done

# Prefer the project virtualenv (Windows layout first, then POSIX), else whatever python is on PATH.
if   [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ];         then PY=.venv/bin/python
else PY=python; fi

echo "== environment =="; "$PY" -m src.environment_info > /dev/null
echo "== unit tests ==";  "$PY" -m pytest -q
echo "== dataset validation =="; "$PY" -m src.validate_data

if [ "$QUICK" -eq 1 ]; then
  echo "(--quick accepted; no benchmark stages exist yet)"
fi
