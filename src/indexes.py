"""FAISS index construction and search-parameter control for Part 2 (and later Part 3).

All indexes use METRIC_L2. Builds run under threads.build_phase(); the caller is responsible
for switching to search_phase() before any timing.

Shared coarse quantizer: IVF-Flat and IVF-PQ with the same nlist reuse one k-means run
(seeded, so a standalone build would produce the same centroids). The reported build time of
every IVF index = coarse k-means time + its own training/adding time, i.e. what a standalone
build would cost; the k-means work is merely not repeated.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import faiss
import numpy as np

from src.config import SEED

HNSW_EF_CONSTRUCTION = 200
PQ_NBITS = 8


@dataclass(frozen=True)
class BuildSpec:
    index_type: str            # "HNSW" | "IVF-Flat" | "IVF-PQ"
    params: tuple              # sorted (name, value) pairs, hashable

    @staticmethod
    def make(index_type: str, **params) -> "BuildSpec":
        return BuildSpec(index_type, tuple(sorted(params.items())))

    @property
    def p(self) -> dict:
        return dict(self.params)

    @property
    def name(self) -> str:
        return self.index_type + "_" + "_".join(f"{k}{v}" for k, v in self.params)

    @property
    def search_param(self) -> str:
        return "efSearch" if self.index_type == "HNSW" else "nprobe"


@dataclass
class BuildResult:
    index: faiss.Index
    build_time_s: float
    timings: dict = field(default_factory=dict)   # breakdown, e.g. coarse_kmeans_s, train_s, add_s


class CoarseQuantizerCache:
    """k-means centroids per nlist, trained once per dataset (seeded)."""

    def __init__(self, x: np.ndarray):
        self.x = x
        self._cache: dict[int, tuple[np.ndarray, float]] = {}

    def get(self, nlist: int) -> tuple[np.ndarray, float]:
        if nlist not in self._cache:
            d = self.x.shape[1]
            q = faiss.IndexFlatL2(d)
            ivf = faiss.IndexIVFFlat(q, d, nlist, faiss.METRIC_L2)
            ivf.cp.seed = SEED
            t0 = time.perf_counter()
            ivf.train(self.x)
            dt = time.perf_counter() - t0
            self._cache[nlist] = (q.reconstruct_n(0, nlist), dt)
        return self._cache[nlist]


def _quantizer(centroids: np.ndarray) -> faiss.IndexFlatL2:
    q = faiss.IndexFlatL2(centroids.shape[1])
    q.add(centroids)
    return q


def build(spec: BuildSpec, x: np.ndarray, coarse: CoarseQuantizerCache | None = None) -> BuildResult:
    d = x.shape[1]
    p = spec.p
    if spec.index_type == "HNSW":
        index = faiss.IndexHNSWFlat(d, int(p["M"]), faiss.METRIC_L2)
        index.hnsw.efConstruction = int(p.get("efC", HNSW_EF_CONSTRUCTION))
        t0 = time.perf_counter()
        index.add(x)
        dt = time.perf_counter() - t0
        return BuildResult(index, dt, {"add_s": dt})

    coarse = coarse or CoarseQuantizerCache(x)
    nlist = int(p["nlist"])
    cents, kmeans_s = coarse.get(nlist)
    q = _quantizer(cents)
    if spec.index_type == "IVF-Flat":
        index = faiss.IndexIVFFlat(q, d, nlist, faiss.METRIC_L2)
    elif spec.index_type == "IVF-PQ":
        m = int(p["m"])
        if d % m:
            raise ValueError(f"PQ m={m} must divide d={d}")
        index = faiss.IndexIVFPQ(q, d, nlist, m, int(p.get("nbits", PQ_NBITS)), faiss.METRIC_L2)
        index.cp.seed = SEED
        index.pq.cp.seed = SEED
    else:
        raise ValueError(spec.index_type)
    # (the FAISS Python wrapper keeps a reference to q for the lifetime of index)
    t0 = time.perf_counter()
    index.train(x)            # coarse quantizer already populated -> only PQ training (if any)
    t1 = time.perf_counter()
    index.add(x)
    t2 = time.perf_counter()
    return BuildResult(index, kmeans_s + (t2 - t0),
                       {"coarse_kmeans_s": kmeans_s, "train_s": t1 - t0, "add_s": t2 - t1})


def add_exact_refine(base: faiss.Index, x: np.ndarray, k_factor: int = 10) -> tuple[faiss.IndexRefineFlat, float]:
    """Wrap a built index with exact re-ranking (Part 3).

    IndexRefineFlat keeps a full float32 copy of x. A search for k asks the base index for
    k * k_factor candidates, recomputes their exact L2 distances and returns the best k.
    The base index is shared, not copied: its nprobe still controls the candidate pool.
    Returns (index, seconds spent copying the raw vectors).
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    # FAISS copies base.ntotal rows from the raw pointer without any bounds check, so the shape
    # must be validated here: a shorter array would be read past its end.
    if x.ndim != 2 or x.shape != (base.ntotal, base.d):
        raise ValueError(f"refine vectors have shape {x.shape}, expected ({base.ntotal}, {base.d})")
    t0 = time.perf_counter()
    refine = faiss.IndexRefineFlat(base, faiss.swig_ptr(x))
    dt = time.perf_counter() - t0
    refine.k_factor = float(k_factor)
    if refine.ntotal != base.ntotal:
        raise RuntimeError(f"refine ntotal {refine.ntotal} != base ntotal {base.ntotal}")
    return refine, dt


def set_search_param(index: faiss.Index, spec: BuildSpec, value: int) -> None:
    if spec.index_type == "HNSW":
        index.hnsw.efSearch = int(value)
    else:
        index.nprobe = int(value)


def index_size_bytes(index: faiss.Index) -> int:
    """Serialised size, measured identically for every index type (decision A3)."""
    return int(faiss.serialize_index(index).size)
