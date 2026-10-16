"""Reading ann-benchmarks HDF5 files, query splitting and GloVe normalisation.

ann-benchmarks files contain four datasets:
  train      (n, d)   float32   vectors to index
  test       (q, d)   float32   queries
  neighbors  (q, 100) int       ids of the true 100 nearest neighbours (in source metric)
  distances  (q, 100) float32   distances to those neighbours (in source metric)
and file attributes such as ``distance`` ("euclidean" / "angular").

Nothing here modifies the original files; normalisation happens on in-memory copies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from src.config import DATASETS, N_TUNE_QUERIES, DatasetSpec

REQUIRED_KEYS = ("train", "test", "neighbors", "distances")


# --------------------------------------------------------------------------- inspection

def inspect_hdf5(path: str | Path) -> dict:
    """Return shapes, dtypes and attributes of every dataset in the file without reading the data."""
    path = Path(path)
    info: dict = {"path": str(path), "file_size_bytes": path.stat().st_size, "datasets": {}}
    with h5py.File(path, "r") as f:
        info["attrs"] = {k: _to_builtin(v) for k, v in f.attrs.items()}
        for key, obj in f.items():
            if isinstance(obj, h5py.Dataset):
                info["datasets"][key] = {
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "chunks": list(obj.chunks) if obj.chunks else None,
                    "compression": obj.compression,
                }
    info["missing_keys"] = [k for k in REQUIRED_KEYS if k not in info["datasets"]]
    return info


def _to_builtin(v):
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def chunked_stats(dset: h5py.Dataset, chunk_rows: int = 100_000) -> dict:
    """Count NaN / inf values and zero-norm rows by streaming the dataset in row blocks.

    Peak extra memory is one block (chunk_rows x d floats), never the whole matrix.
    """
    n = dset.shape[0]
    nan = inf = zero_rows = 0
    norm_min, norm_max = np.inf, -np.inf
    for start in range(0, n, chunk_rows):
        block = np.asarray(dset[start:start + chunk_rows], dtype=np.float32)
        nan += int(np.isnan(block).sum())
        inf += int(np.isinf(block).sum())
        norms = np.linalg.norm(block, axis=1)
        zero_rows += int((norms == 0).sum())
        finite = norms[np.isfinite(norms)]
        if finite.size:
            norm_min = min(norm_min, float(finite.min()))
            norm_max = max(norm_max, float(finite.max()))
    return {"nan_count": nan, "inf_count": inf, "zero_norm_rows": zero_rows,
            "norm_min": norm_min, "norm_max": norm_max}


# --------------------------------------------------------------------------- normalisation

def l2_normalize(x: np.ndarray) -> tuple[np.ndarray, int]:
    """Return a float32 copy of ``x`` with every row scaled to unit L2 norm, plus the zero-row count.

    Rows with norm 0 cannot be normalised; they are left as all-zero vectors (not silently
    dropped) and counted so the caller can record the anomaly.

    Why this is correct for GloVe: for unit vectors ||q - x||^2 = 2 - 2 cos(q, x), so ranking by
    Euclidean distance after normalisation is identical to ranking by angular distance, which is
    the metric the stored ground truth was computed with.
    """
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    zero = norms[:, 0] == 0
    safe = np.where(norms == 0, 1.0, norms).astype(np.float32)
    return x / safe, int(zero.sum())


# --------------------------------------------------------------------------- query split

@dataclass(frozen=True)
class QuerySplit:
    tune: slice
    eval: slice


def split_queries(n_queries: int, n_tune: int = N_TUNE_QUERIES) -> QuerySplit:
    """First ``n_tune`` queries are for tuning, the rest for final evaluation (no overlap, no shuffling)."""
    if not 0 < n_tune < n_queries:
        raise ValueError(f"n_tune={n_tune} must lie strictly between 0 and n_queries={n_queries}")
    return QuerySplit(tune=slice(0, n_tune), eval=slice(n_tune, n_queries))


# --------------------------------------------------------------------------- loading

@dataclass
class Dataset:
    name: str
    train: np.ndarray        # (n, d) float32, normalised if spec.normalize
    queries: np.ndarray      # (q, d) float32, normalised if spec.normalize
    gt_neighbors: np.ndarray  # (q, 100) int64, ground truth from the file
    gt_distances: np.ndarray  # (q, 100) float32, in the *source* metric of the file
    split: QuerySplit
    zero_norm_train: int = 0
    zero_norm_queries: int = 0

    @property
    def tune_queries(self) -> np.ndarray:
        return self.queries[self.split.tune]

    @property
    def eval_queries(self) -> np.ndarray:
        return self.queries[self.split.eval]

    @property
    def tune_gt(self) -> np.ndarray:
        return self.gt_neighbors[self.split.tune]

    @property
    def eval_gt(self) -> np.ndarray:
        return self.gt_neighbors[self.split.eval]


def load_dataset(name_or_spec: str | DatasetSpec, path: str | Path | None = None) -> Dataset:
    """Load a whole ann-benchmarks dataset into RAM (about 0.5 GB for SIFT or GloVe)."""
    spec = DATASETS[name_or_spec] if isinstance(name_or_spec, str) else name_or_spec
    path = Path(path) if path else spec.path
    with h5py.File(path, "r") as f:
        train = np.ascontiguousarray(f["train"][:], dtype=np.float32)
        queries = np.ascontiguousarray(f["test"][:], dtype=np.float32)
        gt_n = f["neighbors"][:].astype(np.int64)
        gt_d = f["distances"][:].astype(np.float32)
    zt = zq = 0
    if spec.normalize:
        train, zt = l2_normalize(train)
        queries, zq = l2_normalize(queries)
    return Dataset(spec.name, train, queries, gt_n, gt_d, split_queries(len(queries)), zt, zq)
