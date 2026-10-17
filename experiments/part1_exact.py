"""Part 1: exact baseline with FAISS IndexFlatL2.

For each dataset:
  1. build IndexFlatL2 under build_phase() (P-core threads) and record build time and size;
  2. exact top-100 for all 10,000 queries (untimed), re-scored in float64 -> reference tau;
     cached in <data>/cache/ for later milestones (never in results/);
  3. compare FAISS exact results and the stored ground truth against tau (full data only);
  4. measure boundary ties with the A2 tolerance;
  5. time exact search under search_phase() (1 thread, pinned P-core):
       - single: one query per call, first N eval queries                      [primary]
       - batch, BLAS path forced (threshold 20), all 5,000 eval queries         [primary batch]
       - batch, FAISS default path (threshold 128000 -> per-query scan), N q.   [explanation]
     and, clearly labelled secondary rows, the same batch paths on 12 P-core threads.
Optionally (--calibrate, SIFT) compare pinned P-core, pinned E-core and unpinned timing.

Usage (from code/):  python -m experiments.part1_exact [--dataset sift] [--quick] [--calibrate]
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager

import faiss
import numpy as np

from src.config import DATASETS, K, METADATA_DIR, RESULTS_DIR, data_dir
from src.data_loader import load_dataset
from src.provenance import run_metadata
from src.recall import (REL_TOL, boundary_tie_stats, recall_at_k, reference_from_candidates,
                        true_sq_distances)
from src.results_io import replace_rows
from src import threads
from src.timing import benchmark_lock, time_search

QUICK_N_BASE = 100_000
REF_WIDTH = 100


@contextmanager
def blas_threshold(n: int):
    """FAISS uses BLAS for exact search only when nq >= this threshold (1.15.1 default: 128000)."""
    old = faiss.cvar.distance_compute_blas_threshold
    faiss.cvar.distance_compute_blas_threshold = n
    try:
        yield old
    finally:
        faiss.cvar.distance_compute_blas_threshold = old


def exact_reference(index, ds):
    """Untimed exact top-100 for all queries, re-scored in float64 and sorted."""
    threads.build_phase()
    with blas_threshold(20):                       # fast path; correctness comes from float64 re-scoring
        _, cand = index.search(ds.queries, REF_WIDTH)
    return cand, *reference_from_candidates(ds.queries, ds.train, cand)


def save_reference(ds, ref_ids, ref_d) -> str:
    cache = data_dir() / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{ds.dataset_id}_exact_top{REF_WIDTH}.npz"
    np.savez(path, ids=ref_ids, dists=ref_d, n_base=len(ds.train))
    return str(path)


def ground_truth_check(ds, faiss_cand, ref_ids, ref_d) -> dict:
    tau = ref_d[:, K - 1]
    f10 = faiss_cand[:, :K]
    f_rec = recall_at_k(f10, true_sq_distances(ds.queries, ds.train, f10), tau)
    out = {"faiss_raw_top10_vs_tau": {"mean_recall": float(f_rec.mean()),
                                      "queries_below_1": int((f_rec < 1).sum())}}
    ref_sets_differ = int(sum(set(a) != set(b) for a, b in zip(f10, ref_ids[:, :K])))
    out["faiss_raw_vs_float64_reference_idset_differs"] = ref_sets_differ
    if ds.gt_neighbors is None:
        out["stored_ground_truth"] = "skipped: base is a subset, stored ground truth does not apply"
        return out
    g10 = ds.gt_neighbors[:, :K]
    g_d = true_sq_distances(ds.queries, ds.train, g10)
    g_rec = recall_at_k(g10, g_d, tau)
    tau_gt = g_d.max(axis=1)
    tau_mismatch = np.abs(tau_gt - tau) > REL_TOL * np.maximum(tau, 1e-12)
    differ = np.array([set(a) != set(b) for a, b in zip(g10, ref_ids[:, :K])])
    out["stored_ground_truth"] = {
        "mean_recall_vs_tau": float(g_rec.mean()),
        "queries_below_1": int((g_rec < 1).sum()),
        "query_ids_below_1": np.flatnonzero(g_rec < 1)[:50].tolist(),
        "tau_mismatch_queries": int(tau_mismatch.sum()),
        "idset_differs_from_reference": int(differ.sum()),
        "idset_differs_but_all_tied (recall 1.0)": int((differ & (g_rec == 1)).sum()),
    }
    return out


def timed_row(base: dict, ds, index, queries, tau, mode, path_thr, state, role, repeats, split_label):
    with blas_threshold(path_thr):
        res = time_search(index, queries, K, repeats, mode)
    rec = recall_at_k(res.ids, true_sq_distances(queries, ds.train, res.ids), tau)
    single = state.omp_threads == 1
    path = "nq=1 scan" if mode == "single" else ("blas" if len(queries) >= path_thr else "per-query scan")
    row = dict(base, search_mode=mode, query_split=split_label, n_queries=len(queries),
               search_params={"blas_threshold": path_thr, "code_path": path},
               recall_at_10=round(float(rec.mean()), 6), thread_check=res.thread_check(single),
               pinned=state.pinned, pin_fallback_reason=state.fallback_reason, notes=role, status="ok")
    row.update(res.as_row())
    row.update(state.as_row("search"))
    print(f"  {mode:6s} {path:15s} omp={state.omp_threads:2d} n={len(queries):5d} "
          f"qps median {res.qps_median:9.1f} (min {min(res.qps):.1f}, max {max(res.qps):.1f}) "
          f"cpu/wall {res.cpu_wall_median:4.1f} recall {rec.mean():.4f}  [{role}]")
    return row


def run_dataset(name: str, quick: bool, n_single: int, repeats: int, sec_repeats: int,
                results_dir, metadata_dir) -> list[dict]:
    meta = run_metadata("p1_exact")
    ds = load_dataset(name, max_base=QUICK_N_BASE if quick else None)
    n, d = ds.train.shape
    print(f"[{ds.dataset_id}] n_base={n} d={d} queries={len(ds.queries)}")

    bst = threads.build_phase()
    t0 = time.perf_counter()
    index = faiss.IndexFlatL2(d)
    index.add(ds.train)
    build_time = time.perf_counter() - t0
    size = len(faiss.serialize_index(index))

    t0 = time.perf_counter()
    cand, ref_ids, ref_d = exact_reference(index, ds)
    ref_time = time.perf_counter() - t0
    cache_path = save_reference(ds, ref_ids, ref_d)
    tau = ref_d[:, K - 1]

    check = ground_truth_check(ds, cand, ref_ids, ref_d)
    ties = {split: boundary_tie_stats(ref_d[sl], K) for split, sl in
            (("all", slice(None)), ("tune", ds.split.tune), ("eval", ds.split.eval))}
    report = {"dataset_id": ds.dataset_id, "n_base": n, "dim": d, "reference_cache": cache_path,
              "reference_width": REF_WIDTH, "reference_time_s": round(ref_time, 2),
              "rel_tol": REL_TOL, "ground_truth_check": check, "boundary_ties": ties,
              "faiss_default_blas_threshold": faiss.cvar.distance_compute_blas_threshold, **meta}
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / f"part1_{ds.dataset_id}_reference.json").write_text(json.dumps(report, indent=2))
    print(f"  ground truth check: {json.dumps(check)}")
    print(f"  boundary ties (all queries): {ties['all']}")

    base = dict(meta, dataset_id=ds.dataset_id, n_base=n, seed="n/a", index_type="Flat",
                build_params={}, build_time_s=round(build_time, 4), index_size_bytes=size,
                bytes_per_vector=round(size / n, 2), **bst.as_row("build"))
    ev = ds.split.eval
    q_eval, tau_eval = ds.queries[ev], tau[ev]
    ns = min(n_single, len(q_eval))
    lab_n, lab_all = f"eval[0:{ns}]", "eval"
    default_thr = faiss.cvar.distance_compute_blas_threshold
    rows = []

    st = threads.search_phase()
    rows.append(timed_row(base, ds, index, q_eval[:ns], tau_eval[:ns], "single", default_thr, st,
                          "primary: one query at a time, 1 thread", repeats, lab_n))
    rows.append(timed_row(base, ds, index, q_eval, tau_eval, "batch", 20, st,
                          "primary batch: BLAS path, 1 thread", repeats, lab_all))
    rows.append(timed_row(base, ds, index, q_eval[:ns], tau_eval[:ns], "batch", default_thr, st,
                          "explanation: FAISS default path (per-query scan), 1 thread", sec_repeats, lab_n))
    st = threads.multi_thread_search()
    rows.append(timed_row(base, ds, index, q_eval, tau_eval, "batch", 20, st,
                          "secondary: BLAS path, P-core threads", sec_repeats, lab_all))
    rows.append(timed_row(base, ds, index, q_eval[:ns], tau_eval[:ns], "batch", default_thr, st,
                          "secondary: FAISS default path, P-core threads", sec_repeats, lab_n))
    threads.search_phase()
    replace_rows(rows, results_dir / "results.csv")
    return rows


def calibrate(quick: bool, n_q: int, repeats: int, results_dir) -> list[dict]:
    """Same single-query workload pinned to a P-core, pinned to an E-core, and unpinned."""
    meta = run_metadata("p1_calibration")
    ds = load_dataset("sift", max_base=QUICK_N_BASE if quick else None)
    n, d = ds.train.shape
    threads.build_phase()
    index = faiss.IndexFlatL2(d)
    index.add(ds.train)
    q = ds.eval_queries[:n_q]
    tau = reference_from_candidates(q, ds.train, index.search(q, 20)[1])[1][:, K - 1]
    topo, _ = threads.cpu_topology()
    slow = [c for c in topo if c.efficiency_class == min(x.efficiency_class for x in topo)]
    modes = [("pinned P-core", threads.search_phase)]
    if slow and slow[0].efficiency_class != max(x.efficiency_class for x in topo):
        modes.append((f"pinned E-core (cpu {slow[0].logical})", lambda: threads.search_phase(slow[0].logical)))
    modes.append(("unpinned, 1 thread", threads.unpinned_single_thread))
    base = dict(meta, dataset_id=ds.dataset_id, n_base=n, seed="n/a", index_type="Flat", build_params={})
    rows = []
    print(f"[calibration] {ds.dataset_id}, {len(q)} single queries x {repeats} repeats")
    for label, phase in modes:
        st = phase()
        row = timed_row(base, ds, index, q, tau, "single", faiss.cvar.distance_compute_blas_threshold,
                        st, f"calibration: {label}", repeats, f"eval[0:{len(q)}]")
        spread = (row["qps_max"] - row["qps_min"]) / row["qps_median"]
        row["notes"] += f"; spread (max-min)/median = {spread:.4f}"
        rows.append(row)
    threads.search_phase()
    replace_rows(rows, results_dir / "results.csv")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), action="append")
    ap.add_argument("--quick", action="store_true", help=f"first {QUICK_N_BASE} base vectors; outputs under results/quick/")
    ap.add_argument("--n-single", type=int, default=500, help="eval queries timed one at a time")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--secondary-repeats", type=int, default=3)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--calibration-queries", type=int, default=200)
    args = ap.parse_args(argv)

    results_dir = RESULTS_DIR / "quick" if args.quick else RESULTS_DIR
    metadata_dir = results_dir / "metadata" if args.quick else METADATA_DIR
    with benchmark_lock(RESULTS_DIR, "part1_exact"):
        if args.calibrate:
            calibrate(args.quick, args.calibration_queries, args.repeats, results_dir)
        for name in args.dataset or ["sift", "glove"]:
            run_dataset(name, args.quick, args.n_single, args.repeats, args.secondary_repeats,
                        results_dir, metadata_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
