"""Query timing (decision A1) and the no-concurrent-benchmarks lock (decision A5).

Single-query mode calls index.search(q[i:i+1], k) for each query in a Python loop, so the
per-call overhead is included and is the same for every index type. Batch mode passes all
queries in one call. Each mode runs one untimed warm-up pass, then `repeats` timed passes over
the same queries; QPS = n_queries / wall time of one pass, and the median is reported.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil

from src.config import RESULTS_DIR
from src.threads import CpuWallTimer

CPU_WALL_LIMIT = 1.3   # single-threaded regions above this are flagged thread_check=FAIL
# time.process_time() advances in 15.625 ms steps on Windows (measured in M5), so cpu/wall says
# nothing about a pass shorter than a few ticks; such checks are reported as INCONCLUSIVE.
MIN_CHECK_WALL_S = 0.0625


@dataclass
class TimingResult:
    mode: str
    n_queries: int
    repeats: int
    qps: list[float]
    cpu_wall: list[float]
    ids: np.ndarray                      # ids from the last timed pass

    @property
    def qps_median(self) -> float:
        return statistics.median(self.qps)

    @property
    def cpu_wall_median(self) -> float:
        return statistics.median(self.cpu_wall)

    def thread_check(self, expect_single: bool) -> str:
        if not expect_single:
            return "n/a"
        if self.n_queries / self.qps_median < MIN_CHECK_WALL_S:      # median wall time of one pass
            return "INCONCLUSIVE"
        return "PASS" if self.cpu_wall_median <= CPU_WALL_LIMIT else "FAIL"

    def as_row(self) -> dict:
        return {"search_mode": self.mode, "n_timed_queries": self.n_queries, "n_repeats": self.repeats,
                "qps_median": round(self.qps_median, 3), "qps_min": round(min(self.qps), 3),
                "qps_max": round(max(self.qps), 3), "qps_all": json.dumps([round(q, 3) for q in self.qps]),
                "search_cpu_wall": round(self.cpu_wall_median, 3),
                "latency_ms_median": round(1000.0 / self.qps_median, 4)}


def _single_pass(index, queries: np.ndarray, k: int) -> np.ndarray:
    out = np.empty((len(queries), k), dtype=np.int64)
    for i in range(len(queries)):
        _, I = index.search(queries[i:i + 1], k)
        out[i] = I[0]
    return out


def _batch_pass(index, queries: np.ndarray, k: int) -> np.ndarray:
    return index.search(queries, k)[1]


def time_search(index, queries: np.ndarray, k: int = 10, repeats: int = 5, mode: str = "single",
                warmup: bool = True) -> TimingResult:
    if mode not in ("single", "batch"):
        raise ValueError(mode)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    run = _single_pass if mode == "single" else _batch_pass
    if warmup:
        run(index, queries, k)
    qps, ratio, ids = [], [], None
    for _ in range(repeats):
        with CpuWallTimer() as t:
            ids = run(index, queries, k)
        qps.append(len(queries) / t.wall)
        ratio.append(t.ratio)
    return TimingResult(mode, len(queries), repeats, qps, ratio, ids)


# --------------------------------------------------------------------------- lock

class BenchmarkLockError(RuntimeError):
    pass


@contextmanager
def benchmark_lock(results_dir: Path = RESULTS_DIR, label: str = ""):
    """Refuse to run if another benchmark holds results/.benchmark.lock.

    A lock whose PID is no longer alive is treated as stale and replaced (and reported).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / ".benchmark.lock"
    note = ""
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                info = json.loads(path.read_text())
                alive = psutil.pid_exists(int(info.get("pid", -1)))
            except (ValueError, OSError):
                info, alive = {}, False
            if alive:
                raise BenchmarkLockError(f"another benchmark is running: {info} (lock {path})")
            note = f"removed stale lock {info}"
            path.unlink(missing_ok=True)
    else:
        raise BenchmarkLockError(f"could not acquire {path}")
    with os.fdopen(fd, "w") as f:
        json.dump({"pid": os.getpid(), "label": label,
                   "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f)
    if note:
        print(f"[lock] {note}")
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)
