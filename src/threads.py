"""Thread and CPU-affinity control for index building and query timing (decisions A1, A5).

FAISS has two independent thread pools (verified in design revision B):
  * OpenMP, controlled by faiss.omp_set_num_threads;
  * the OpenBLAS bundled with faiss-cpu (default 20 threads on the dev machine), used for
    batched distance computations and k-means; controlled here through ctypes.
Every phase sets affinity, OpenMP and OpenBLAS, then reads all three back and raises if
any differs. Nothing silently continues with the wrong configuration.

Phases:
  build_phase()   affinity = all fastest-class cores (P-cores), OpenMP = OpenBLAS = their count
  search_phase()  affinity = one logical CPU of one P-core,      OpenMP = OpenBLAS = 1
"""

from __future__ import annotations

import ctypes
import glob
import os
import struct
import sys
import time
from dataclasses import asdict, dataclass, field

import faiss
import psutil


# --------------------------------------------------------------------------- topology

@dataclass(frozen=True)
class LogicalCpu:
    logical: int
    core: int
    efficiency_class: int


def cpu_topology() -> tuple[list[LogicalCpu], str]:
    """Return the logical CPUs with core id and efficiency class, plus the source used.

    Windows: GetSystemCpuSetInformation (higher EfficiencyClass = faster core).
    Elsewhere: every CPU gets class 0 (no hybrid information), which makes all cores "P-cores".
    """
    if sys.platform == "win32":
        try:
            return _windows_cpu_sets(), "GetSystemCpuSetInformation"
        except OSError:
            pass
    n = psutil.cpu_count(logical=True) or 1
    return [LogicalCpu(i, i, 0) for i in range(n)], "fallback:uniform"


def _windows_cpu_sets() -> list[LogicalCpu]:
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    need = ctypes.c_ulong(0)
    k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(need), None, 0)
    buf = ctypes.create_string_buffer(need.value)
    if not k32.GetSystemCpuSetInformation(buf, need, ctypes.byref(need), None, 0):
        raise OSError(ctypes.get_last_error(), "GetSystemCpuSetInformation failed")
    cpus, off = [], 0
    while off < need.value:
        size, typ = struct.unpack_from("<II", buf, off)
        if typ == 0:  # CpuSetInformation
            _id, group, lp, core, _llc, _numa, eff = struct.unpack_from("<IHBBBBB", buf, off + 8)
            if group != 0:
                raise OSError(0, "processor groups > 0 not supported")
            cpus.append(LogicalCpu(lp, core, eff))
        off += size
    if not cpus:
        raise OSError(0, "no CPU sets reported")
    return sorted(cpus, key=lambda c: c.logical)


def fast_cpus(topo: list[LogicalCpu]) -> list[LogicalCpu]:
    best = max(c.efficiency_class for c in topo)
    return [c for c in topo if c.efficiency_class == best]


def search_cpu(topo: list[LogicalCpu]) -> LogicalCpu:
    """First logical CPU of the highest-numbered fast core (core 0 tends to get more OS interrupts)."""
    fast = fast_cpus(topo)
    last_core = max(c.core for c in fast)
    return min((c for c in fast if c.core == last_core), key=lambda c: c.logical)


# --------------------------------------------------------------------------- OpenBLAS inside faiss

class _FaissBlas:
    """ctypes handle on the OpenBLAS shipped with faiss-cpu, if it can be found."""

    def __init__(self):
        self.lib = None
        self.path = None
        libs_dir = os.path.join(os.path.dirname(faiss.__file__), os.pardir, "faiss_cpu.libs")
        pats = ("libopenblas*.dll",) if sys.platform == "win32" else ("libopenblas*.so*", "libopenblas*.dylib")
        for pat in pats:
            hits = glob.glob(os.path.join(libs_dir, pat))
            if hits:
                try:
                    lib = ctypes.CDLL(os.path.abspath(hits[0]))
                    lib.openblas_get_num_threads.restype = ctypes.c_int
                    lib.openblas_set_num_threads.argtypes = [ctypes.c_int]
                    self.lib, self.path = lib, os.path.abspath(hits[0])
                except (OSError, AttributeError):
                    pass
                break

    @property
    def available(self) -> bool:
        return self.lib is not None

    def get(self) -> int | None:
        return int(self.lib.openblas_get_num_threads()) if self.lib else None

    def set(self, n: int) -> None:
        if self.lib:
            self.lib.openblas_set_num_threads(int(n))

    def config(self) -> str | None:
        if not self.lib:
            return None
        try:
            self.lib.openblas_get_config.restype = ctypes.c_char_p
            return self.lib.openblas_get_config().decode()
        except AttributeError:
            return None


_BLAS = None


def faiss_blas() -> _FaissBlas:
    global _BLAS
    if _BLAS is None:
        _BLAS = _FaissBlas()
    return _BLAS


# --------------------------------------------------------------------------- phases

class ThreadConfigError(RuntimeError):
    pass


@dataclass
class ThreadState:
    phase: str
    affinity: list[int]
    omp_threads: int
    blas_threads: int | None
    pinned: bool
    efficiency_classes: list[int]
    topology_source: str
    blas_available: bool
    fallback_reason: str = ""
    extra: dict = field(default_factory=dict)

    def as_row(self, prefix: str) -> dict:
        return {
            f"{prefix}_affinity": " ".join(map(str, self.affinity)),
            f"{prefix}_omp": self.omp_threads,
            f"{prefix}_blas": self.blas_threads if self.blas_threads is not None else "unavailable",
        }


def _apply(phase: str, cpus: list[int], n_threads: int, topo, source: str, pinned: bool,
           fallback_reason: str = "") -> ThreadState:
    proc = psutil.Process()
    reason = fallback_reason
    affinity_ok = hasattr(proc, "cpu_affinity")
    if affinity_ok:
        try:
            proc.cpu_affinity(cpus)
            got = sorted(proc.cpu_affinity())
            if got != sorted(cpus):
                raise ThreadConfigError(f"{phase}: affinity readback {got} != requested {sorted(cpus)}")
        except (psutil.Error, OSError) as e:          # setting affinity failed: run unpinned, record why
            affinity_ok, pinned = False, False
            reason = f"affinity unavailable: {e}"
    else:
        pinned, reason = False, "psutil has no cpu_affinity on this platform"

    faiss.omp_set_num_threads(n_threads)
    if faiss.omp_get_max_threads() != n_threads:
        raise ThreadConfigError(f"{phase}: OpenMP readback {faiss.omp_get_max_threads()} != {n_threads}")
    blas = faiss_blas()
    blas.set(n_threads)
    if blas.available and blas.get() != n_threads:
        raise ThreadConfigError(f"{phase}: OpenBLAS readback {blas.get()} != {n_threads}")

    affinity = sorted(proc.cpu_affinity()) if affinity_ok else list(range(psutil.cpu_count() or 1))
    by_id = {c.logical: c for c in topo}
    return ThreadState(phase=phase, affinity=affinity, omp_threads=faiss.omp_get_max_threads(),
                       blas_threads=blas.get(), pinned=pinned,
                       efficiency_classes=sorted({by_id[c].efficiency_class for c in affinity if c in by_id}),
                       topology_source=source, blas_available=blas.available, fallback_reason=reason)


def build_phase() -> ThreadState:
    topo, source = cpu_topology()
    cpus = [c.logical for c in fast_cpus(topo)]
    return _apply("build", cpus, len(cpus), topo, source, pinned=True)


def search_phase(cpu: int | None = None) -> ThreadState:
    """Single-threaded search pinned to one logical CPU (default: derived P-core CPU)."""
    topo, source = cpu_topology()
    chosen = search_cpu(topo).logical if cpu is None else cpu
    return _apply("search", [chosen], 1, topo, source, pinned=True)


def unpinned_single_thread() -> ThreadState:
    """Single-threaded search allowed on every logical CPU (calibration control)."""
    topo, source = cpu_topology()
    return _apply("search_unpinned", [c.logical for c in topo], 1, topo, source, pinned=False)


def multi_thread_search() -> ThreadState:
    """Secondary, clearly labelled multi-threaded search on all P-cores (not used for QPS tables)."""
    st = build_phase()
    st.phase = "search_multithread"
    return st


def describe() -> dict:
    """Static facts for the environment record (does not change the current configuration)."""
    topo, source = cpu_topology()
    blas = faiss_blas()
    return {
        "topology_source": source,
        "logical_cpus": [asdict(c) for c in topo],
        "fast_logical_cpus": [c.logical for c in fast_cpus(topo)],
        "search_cpu": asdict(search_cpu(topo)),
        "faiss_openblas_path": blas.path,
        "faiss_openblas_config": blas.config(),
        "faiss_openblas_threads_now": blas.get(),
        "faiss_omp_threads_now": faiss.omp_get_max_threads(),
    }


# --------------------------------------------------------------------------- measuring

class CpuWallTimer:
    """Wall time and process CPU time over a region; cpu/wall ~ 1 means one busy thread.

    time.process_time counts CPU time of all threads of the process.
    """

    def __enter__(self):
        self.c0, self.t0 = time.process_time(), time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.wall = time.perf_counter() - self.t0
        self.cpu = time.process_time() - self.c0
        return False

    @property
    def ratio(self) -> float:
        return self.cpu / self.wall if self.wall > 0 else float("nan")
