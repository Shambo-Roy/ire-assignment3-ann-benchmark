#!/usr/bin/env bash
# One-command reproduction. Usage:  bash run_all.sh [--quick]
#
# Outputs go to ../results (or ../results/quick with --quick); see README.md.
# STATUS: partial. Implemented stages: environment, tests, dataset validation, Parts 1-4.
# Stages run strictly one after another (no concurrent benchmarks; see decision A5).
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
  echo "== Part 1: exact baseline (quick: first 100,000 base vectors) =="
  "$PY" -u -m experiments.part1_exact --quick --calibrate --n-single 200 --repeats 3 \
        --secondary-repeats 2 --calibration-queries 100
  echo "== Part 2: ANN benchmarks (quick grid) =="
  "$PY" -u -m experiments.part2_ann --quick --n-time-tune 200 --repeats-tune 2 \
        --n-time-eval 200 --repeats-eval 3
  "$PY" -m experiments.part2_report --quick
  echo "== Part 3: compression (quick; hypothesis guard first) =="
  "$PY" -u -m experiments.part3_compression --quick --n-time-tune 200 --repeats-tune 2 \
        --n-time-eval 200 --repeats-eval 3
  "$PY" -m experiments.part3_report --quick
  echo "== Part 4: dimensionality (quick; hypothesis guard first) =="
  "$PY" -u -m experiments.part4_dimensionality --quick --repeats 3
  "$PY" -m experiments.part4_report --quick
else
  echo "== Part 1: exact baseline =="
  "$PY" -u -m experiments.part1_exact --calibrate
  echo "== Part 2: ANN benchmarks (full grid; long) =="
  "$PY" -u -m experiments.part2_ann
  "$PY" -m experiments.part2_report
  "$PY" -u -m experiments.build_scaling
  echo "== Part 3: compression (hypothesis guard first) =="
  "$PY" -u -m experiments.part3_compression
  "$PY" -m experiments.part3_report
  echo "== Part 4: dimensionality (hypothesis guard first) =="
  "$PY" -u -m experiments.part4_dimensionality
  "$PY" -m experiments.part4_report
fi
