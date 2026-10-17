"""Distance-based recall@k (decision A2 in IMPLEMENTATION_LOG.md).

A returned id is correct when its *true* distance to the query is no larger than the true
k-th nearest-neighbour distance tau (plus a tiny relative tolerance). Any vector tied with the
k-th neighbour therefore counts as correct, even if its id differs from the reference list.

True distances are squared L2 computed in float64 from the stored float32 vectors, for both
the reference and the candidates, so equal distances compare equal. FAISS's own float32
distances are never used for correctness.
"""

from __future__ import annotations

import numpy as np

REL_TOL = 1e-6
# Floor for the tolerance scale, so tau == 0 (query identical to a base vector) still works.
ABS_FLOOR = 1e-12


def true_sq_distances(queries: np.ndarray, base: np.ndarray, ids: np.ndarray,
                      chunk: int = 256) -> np.ndarray:
    """Exact squared L2 distances, float64, between each query and its candidate ids.

    ids may contain -1 (FAISS padding); those positions get +inf.
    Work is chunked over queries so at most chunk x width x d floats are gathered at once.
    """
    ids = np.asarray(ids)
    nq, width = ids.shape
    out = np.full((nq, width), np.inf, dtype=np.float64)
    for s in range(0, nq, chunk):
        blk = ids[s:s + chunk]
        valid = blk >= 0
        vecs = base[np.where(valid, blk, 0)].astype(np.float64)          # (b, width, d)
        diff = vecs - queries[s:s + chunk, None, :].astype(np.float64)
        d = np.einsum("bwd,bwd->bw", diff, diff)
        out[s:s + chunk] = np.where(valid, d, np.inf)
    return out


def reference_from_candidates(queries: np.ndarray, base: np.ndarray, cand_ids: np.ndarray):
    """Re-score exact-search candidates in float64 and sort them nearest first.

    Returns (ids, dists), both (nq, width), ordered by the float64 distance (ties by id).
    Used to turn FAISS IndexFlatL2 output into a reference whose column k-1 is tau.
    """
    d = true_sq_distances(queries, base, cand_ids)
    order = _row_lexsort(d, cand_ids)
    return np.take_along_axis(cand_ids, order, 1), np.take_along_axis(d, order, 1)


def _row_lexsort(d: np.ndarray, ids: np.ndarray) -> np.ndarray:
    # Sort by distance, then by id, so the order is deterministic among exact ties.
    order = np.argsort(ids, axis=1, kind="stable")
    d_sorted = np.take_along_axis(d, order, 1)
    second = np.argsort(d_sorted, axis=1, kind="stable")
    return np.take_along_axis(order, second, 1)


def tolerance_threshold(tau: np.ndarray, rel_tol: float = REL_TOL) -> np.ndarray:
    tau = np.asarray(tau, dtype=np.float64)
    return tau + rel_tol * np.maximum(tau, ABS_FLOOR)


def recall_at_k(ids: np.ndarray, dists: np.ndarray, tau: np.ndarray, k: int = 10,
                rel_tol: float = REL_TOL) -> np.ndarray:
    """Per-query recall@k.

    ids:   (nq, w) returned ids; only the first k columns are scored; w < k is allowed.
    dists: (nq, w) true float64 distances of those ids (see true_sq_distances).
    tau:   (nq,) true k-th nearest distance from the exact reference.
    A position is correct if its id is >= 0, is the first occurrence of that id in the row,
    and its true distance <= tau + rel_tol * max(tau, ABS_FLOOR). The denominator is always k.
    """
    ids = np.asarray(ids)[:, :k]
    dists = np.asarray(dists, dtype=np.float64)[:, :k]
    if ids.shape != dists.shape:
        raise ValueError(f"ids {ids.shape} and dists {dists.shape} differ")
    valid = ids >= 0
    # first occurrence of each id within its row
    order = np.argsort(ids, axis=1, kind="stable")
    s = np.take_along_axis(ids, order, 1)
    first_sorted = np.ones_like(s, dtype=bool)
    first_sorted[:, 1:] = s[:, 1:] != s[:, :-1]
    first = np.empty_like(first_sorted)
    np.put_along_axis(first, order, first_sorted, 1)
    thr = tolerance_threshold(tau, rel_tol)[:, None]
    correct = valid & first & (dists <= thr)
    return correct.sum(axis=1) / k


def mean_recall(*args, **kwargs) -> float:
    return float(recall_at_k(*args, **kwargs).mean())


def boundary_tie_stats(ref_dists: np.ndarray, k: int = 10, rel_tol: float = REL_TOL) -> dict:
    """How often the reference has another vector tied with the k-th neighbour.

    ref_dists: (nq, w) sorted float64 reference distances with w > k.
    A query has a boundary tie when its (k+1)-th distance is within tolerance of the k-th.
    Also reports the smallest non-zero relative gap, to show how far the tolerance is from
    separating genuinely different distances.
    """
    if ref_dists.shape[1] <= k:
        raise ValueError("need more than k reference columns to detect ties")
    tau = ref_dists[:, k - 1]
    thr = tolerance_threshold(tau, rel_tol)
    n_within = (ref_dists[:, k:] <= thr[:, None]).sum(axis=1)       # extra vectors tied at the boundary
    gap = (ref_dists[:, k] - tau) / np.maximum(tau, ABS_FLOOR)       # relative gap, rank k+1 vs rank k
    positive = gap[gap > 0]
    # Sensitivity: how many queries count as tied under other tolerances (0 = exact equality).
    by_tol = {f"{t:g}": int((gap <= t).sum()) for t in (0.0, 1e-7, 1e-6, 1e-5, 1e-4)}
    return {
        "n_queries": int(len(tau)),
        "queries_with_boundary_tie": int((n_within > 0).sum()),
        "max_extra_tied_vectors": int(n_within.max()) if len(tau) else 0,
        "queries_with_exact_equal_k_kplus1": int((gap == 0).sum()),
        "queries_tied_by_tolerance": by_tol,
        "min_positive_relative_gap": float(positive.min()) if positive.size else None,
        "min_relative_gap_above_tol": float(gap[gap > rel_tol].min()) if (gap > rel_tol).any() else None,
        "rel_tol": rel_tol,
        "tie_detection_limited_to_width": int(ref_dists.shape[1]),
    }
