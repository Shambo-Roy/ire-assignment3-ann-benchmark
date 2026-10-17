"""Part 2: benchmark HNSW, IVF-Flat and IVF-PQ (FAISS) on SIFT and GloVe.

Protocol (tune on the first 5,000 queries, report on the last 5,000):
  1. For each build configuration: build under build_phase() (P-core threads), record build
     time and serialised size.
  2. TUNE: sweep the search parameter. For every setting,
       recall@10 on all 5,000 tune queries (batch search, untimed; same ids as single-query search,
       which is checked on the timed subset), and
       QPS = single-query, single-thread, pinned P-core timing on the first N tune queries.
     A sweep stops early when tune recall >= STOP_RECALL, when QPS drops below the exact-search
     QPS from Part 1 (an ANN slower than exact search is pointless), when recall has plateaued,
     or when nprobe would exceed nlist.
  3. SELECT on tune rows only: the Pareto front of (recall, QPS) per index type, which includes
     the fastest configuration reaching recall 0.90 and 0.95 (when reachable).
  4. EVAL: re-measure only the selected configurations on the eval queries: recall on all 5,000,
     QPS on the first N eval queries with R repeats. Reported numbers come from these rows only.

Usage (from code/):
  python -m experiments.part2_ann [--dataset sift] [--index HNSW] [--quick]
"""

from __future__ import annotations

import argparse
import gc
import json
import time

import faiss
import numpy as np

from src import threads
from src.config import DATASETS, K, METADATA_DIR, RESULTS_DIR, SEED, data_dir
from src.data_loader import load_dataset
from src.indexes import (HNSW_EF_CONSTRUCTION, PQ_NBITS, BuildSpec, CoarseQuantizerCache, build,
                         index_size_bytes, set_search_param)
from src.pareto import pareto_front, select_for_target
from src.provenance import run_metadata
from src.recall import recall_at_k, true_sq_distances
from src.results_io import read_rows, replace_rows
from src.timing import benchmark_lock, time_search

QUICK_N_BASE = 100_000
TARGETS = (0.90, 0.95)
STOP_RECALL = 0.995
PLATEAU_GAIN = 0.002          # stop when two consecutive steps each gain less than this
INDEX_TYPES = ("HNSW", "IVF-Flat", "IVF-PQ")
KEY = ("experiment_id", "dataset_id", "n_base", "index_type")

NPROBE = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
EF = [10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
PQ_M = {"sift": (16, 32, 64), "glove": (10, 25, 50)}

GRIDS = {
    "full": {
        "HNSW": ([{"M": 8}, {"M": 16}, {"M": 32}], EF),
        "IVF-Flat": ([{"nlist": 1024}, {"nlist": 4096}, {"nlist": 16384}], NPROBE),
        "IVF-PQ": (lambda ds: [{"nlist": nl, "m": m} for nl in (1024, 4096) for m in PQ_M[ds]], NPROBE),
    },
    "quick": {
        "HNSW": ([{"M": 16}], [10, 20, 40, 80, 160, 320]),
        "IVF-Flat": ([{"nlist": 256}], [1, 2, 4, 8, 16, 32, 64, 128]),
        "IVF-PQ": (lambda ds: [{"nlist": 256, "m": PQ_M[ds][0]}], [1, 2, 4, 8, 16, 32, 64, 128]),
    },
}


def build_specs(grid: str, index_type: str, dataset: str) -> tuple[list[BuildSpec], list[int]]:
    builds, sweep = GRIDS[grid][index_type]
    builds = builds(dataset) if callable(builds) else builds
    specs = []
    for b in builds:
        if index_type == "HNSW":
            b = dict(b, efC=HNSW_EF_CONSTRUCTION)
        if index_type == "IVF-PQ":
            b = dict(b, nbits=PQ_NBITS)
        specs.append(BuildSpec.make(index_type, **b))
    return specs, sweep


def load_reference_tau(ds) -> np.ndarray:
    """tau (true 10th-NN distance) for all queries, from the Part 1 cache (recomputed if absent)."""
    path = data_dir() / "cache" / f"{ds.dataset_id}_exact_top100.npz"
    if not path.exists():
        from experiments.part1_exact import exact_reference, save_reference
        index = faiss.IndexFlatL2(ds.train.shape[1])
        index.add(ds.train)
        _, ref_ids, ref_d = exact_reference(index, ds)
        save_reference(ds, ref_ids, ref_d)
        del index
    z = np.load(path)
    if int(z["n_base"]) != len(ds.train):
        raise RuntimeError(f"reference cache {path} is for n_base={int(z['n_base'])}")
    return z["dists"][:, K - 1]


def exact_qps(dataset_id: str, results_path) -> float | None:
    for r in read_rows(results_path):
        if (r["experiment_id"] == "p1_exact" and r["dataset_id"] == dataset_id
                and r["notes"].startswith("primary: one query") and r["reportable"] == "True"):
            return float(r["qps_median"])
    return None


class Measurer:
    def __init__(self, ds, tau, n_time: int, repeats: int):
        self.ds, self.tau, self.n_time, self.repeats = ds, tau, n_time, repeats

    def measure(self, index, queries, tau, split_label) -> dict:
        threads.build_phase()                                  # untimed: recall on the whole split
        ids_all = index.search(queries, K)[1]
        rec = recall_at_k(ids_all, true_sq_distances(queries, self.ds.train, ids_all), tau)
        st = threads.search_phase()                            # timed: single query, 1 thread, pinned
        n = min(self.n_time, len(queries))
        t = time_search(index, queries[:n], K, self.repeats, "single")
        match = bool(np.array_equal(t.ids, ids_all[:n]))
        row = {"query_split": split_label, "n_queries": len(queries), "recall_at_10": round(float(rec.mean()), 6),
               "thread_check": t.thread_check(True), "pinned": st.pinned, "pin_fallback_reason": st.fallback_reason,
               "_single_batch_ids_match": match}
        row.update(t.as_row())
        row.update(st.as_row("search"))
        return row


def run_index_type(ds, index_type, grid, tau, ex_qps, args, results_path, build_log) -> list[dict]:
    specs, sweep = build_specs(grid, index_type, ds.name)
    tune_q, tau_tune = ds.queries[ds.split.tune], tau[ds.split.tune]
    eval_q, tau_eval = ds.queries[ds.split.eval], tau[ds.split.eval]
    tuner = Measurer(ds, tau, args.n_time_tune, args.repeats_tune)
    evaler = Measurer(ds, tau, args.n_time_eval, args.repeats_eval)
    coarse = CoarseQuantizerCache(ds.train)
    meta_tune = run_metadata("p2_ann")
    n = len(ds.train)
    built, tune_rows = {}, []

    for spec in specs:
        bst = threads.build_phase()
        try:
            res = build(spec, ds.train, coarse)
        except Exception as e:                                  # never skip silently
            tune_rows.append(dict(meta_tune, dataset_id=ds.dataset_id, n_base=n, index_type=index_type,
                                  build_params=spec.p, status="failed", error=repr(e), query_split="tune"))
            print(f"  BUILD FAILED {spec.name}: {e!r}")
            continue
        size = index_size_bytes(res.index)
        built[spec] = res
        build_log[spec.name] = {"build_time_s": res.build_time_s, **res.timings, "index_size_bytes": size,
                                "build_threads": bst.as_row("build")}
        print(f"  built {spec.name}: {res.build_time_s:.1f} s, {size / 1e6:.1f} MB, "
              f"{size / n:.1f} B/vec  {json.dumps({k: round(v, 1) for k, v in res.timings.items()})}")
        base = dict(meta_tune, dataset_id=ds.dataset_id, n_base=n, index_type=index_type,
                    build_params=spec.p, build_time_s=round(res.build_time_s, 3), index_size_bytes=size,
                    bytes_per_vector=round(size / n, 2), seed=SEED if index_type != "HNSW" else "faiss-default",
                    status="ok", **bst.as_row("build"))
        history = []
        for v in sweep:
            if index_type != "HNSW" and v > spec.p["nlist"]:
                break
            set_search_param(res.index, spec, v)
            m = tuner.measure(res.index, tune_q, tau_tune, "tune")
            match = m.pop("_single_batch_ids_match")
            row = dict(base, **m, search_params={spec.search_param: v},
                       notes=f"tuning sweep; single==batch ids on timed subset: {match}")
            tune_rows.append(row)
            history.append(row["recall_at_10"])
            print(f"    tune {spec.search_param}={v:5d} recall {row['recall_at_10']:.4f} "
                  f"qps {row['qps_median']:9.1f} cpu/wall {row['search_cpu_wall']:.2f}{'' if match else '  IDS MISMATCH'}")
            if row["recall_at_10"] >= STOP_RECALL:
                break
            if ex_qps and row["qps_median"] < ex_qps:
                print(f"    stop: QPS below exact search ({ex_qps:.1f})")
                break
            if len(history) >= 3 and history[-1] - history[-2] < PLATEAU_GAIN and history[-2] - history[-3] < PLATEAU_GAIN:
                print("    stop: recall plateau")
                break

    # ---- select on tune rows only
    ok = [r for r in tune_rows if r.get("status") == "ok" and "recall_at_10" in r]
    pts = [{"recall": r["recall_at_10"], "qps": r["qps_median"], "row": r} for r in ok]
    front = pareto_front(pts)
    chosen = {id(p["row"]): ["pareto"] for p in front}
    for t in TARGETS:
        sel = select_for_target(pts, t)
        if sel is not None:
            chosen.setdefault(id(sel["row"]), ["pareto"]).append(f"target{t:.2f}")
    for r in ok:
        if id(r) in chosen:
            r["notes"] += "; selected for eval: " + ",".join(chosen[id(r)])

    # ---- eval the selected configurations
    meta_eval = run_metadata("p2_ann")
    eval_rows = []
    by_name = {s.name: s for s in built}
    for r in ok:
        if id(r) not in chosen:
            continue
        spec = by_name[BuildSpec.make(index_type, **r["build_params"]).name]
        v = r["search_params"][spec.search_param]
        set_search_param(built[spec].index, spec, v)
        m = evaler.measure(built[spec].index, eval_q, tau_eval, "eval")
        match = m.pop("_single_batch_ids_match")
        keep = {k: r[k] for k in ("dataset_id", "n_base", "index_type", "build_params", "build_time_s",
                                  "index_size_bytes", "bytes_per_vector", "seed", "build_affinity",
                                  "build_omp", "build_blas", "status")}
        row = dict(meta_eval, **keep, **m, search_params=r["search_params"],
                   notes=f"eval of tune-selected config ({','.join(chosen[id(r)])}); "
                         f"single==batch ids on timed subset: {match}")
        eval_rows.append(row)
        print(f"    EVAL {spec.name} {spec.search_param}={v:5d} recall {row['recall_at_10']:.4f} "
              f"qps {row['qps_median']:9.1f} [{','.join(chosen[id(r)])}]{'' if match else '  IDS MISMATCH'}")

    rows = tune_rows + eval_rows
    replace_rows(rows, results_path, key=KEY)
    del built, coarse
    gc.collect()
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), action="append")
    ap.add_argument("--index", choices=INDEX_TYPES, action="append")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n-time-tune", type=int, default=1000)
    ap.add_argument("--repeats-tune", type=int, default=3)
    ap.add_argument("--n-time-eval", type=int, default=1000)
    ap.add_argument("--repeats-eval", type=int, default=5)
    args = ap.parse_args(argv)
    grid = "quick" if args.quick else "full"
    results_dir = RESULTS_DIR / "quick" if args.quick else RESULTS_DIR
    metadata_dir = results_dir / "metadata" if args.quick else METADATA_DIR
    metadata_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "results.csv"

    with benchmark_lock(RESULTS_DIR, "part2_ann"):
        for name in args.dataset or ["sift", "glove"]:
            t0 = time.perf_counter()
            ds = load_dataset(name, max_base=QUICK_N_BASE if args.quick else None)
            tau = load_reference_tau(ds)
            ex = exact_qps(ds.dataset_id, results_path)
            print(f"[{ds.dataset_id}] n={len(ds.train)} d={ds.train.shape[1]} grid={grid} exact single QPS={ex}")
            log_path = metadata_dir / f"part2_{ds.dataset_id}_builds.json"
            build_log = json.loads(log_path.read_text()) if log_path.exists() else {}
            for it in args.index or INDEX_TYPES:
                t1 = time.perf_counter()
                print(f" == {it}")
                run_index_type(ds, it, grid, tau, ex, args, results_path, build_log)
                log_path.write_text(json.dumps(build_log, indent=2))
                print(f" == {it} done in {(time.perf_counter() - t1) / 60:.1f} min")
            print(f"[{ds.dataset_id}] done in {(time.perf_counter() - t0) / 60:.1f} min")
            del ds
            gc.collect()
    threads.search_phase()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
