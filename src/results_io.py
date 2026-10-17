"""results/results.csv: one row per benchmark run behind a reported number (decision A6 schema).

Re-running a driver replaces only the rows with the same (experiment_id, dataset_id, n_base),
so the file stays reproducible and a rerun never duplicates rows.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from src.config import RESULTS_DIR

COLUMNS = [
    # provenance
    "run_id", "experiment_id", "timestamp_utc", "git_commit", "git_dirty", "reportable",
    "hypothesis_commit", "hypothesis_sha256",
    # input
    "dataset_id", "n_base", "query_split", "n_queries", "seed",
    # configuration
    "index_type", "build_params", "search_mode", "search_params",
    # threads
    "build_affinity", "build_omp", "build_blas", "search_affinity", "search_omp", "search_blas",
    "pinned", "pin_fallback_reason", "search_cpu_wall", "thread_check",
    # metrics
    "recall_at_10", "n_timed_queries", "n_repeats", "qps_median", "qps_min", "qps_max", "qps_all",
    "latency_ms_median", "build_time_s", "index_size_bytes", "bytes_per_vector",
    # outcome
    "status", "error", "notes",
]


def results_csv(results_dir: Path = RESULTS_DIR) -> Path:
    return results_dir / "results.csv"


def _cell(v):
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True)
    return "" if v is None else v


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def replace_rows(rows: list[dict], path: Path | None = None) -> Path:
    """Write rows, replacing earlier rows of the same (experiment_id, dataset_id, n_base)."""
    path = path or results_csv()
    unknown = {k for r in rows for k in r} - set(COLUMNS)
    if unknown:
        raise KeyError(f"unknown result columns: {sorted(unknown)}")
    keys = {(str(r["experiment_id"]), str(r["dataset_id"]), str(r["n_base"])) for r in rows}
    kept = [r for r in read_rows(path)
            if (r["experiment_id"], r["dataset_id"], r["n_base"]) not in keys]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in kept:
            w.writerow({c: r.get(c, "") for c in COLUMNS})
        for r in rows:
            w.writerow({c: _cell(r.get(c)) for c in COLUMNS})
    tmp.replace(path)
    return path
