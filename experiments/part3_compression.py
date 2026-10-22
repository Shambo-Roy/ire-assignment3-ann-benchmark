"""Part 3: compression on SIFT, IVF-PQ vs IVF-PQ + exact re-ranking of the top 100.

Tests the author's hypotheses 3.1-3.3 (hypothesis.md, committed in 55016fe). The hypothesis guard
runs FIRST and the script refuses to start (exit 3) unless the Part 3 section is committed and
unchanged. Rows record hypothesis_commit and hypothesis_sha256.

Design (one property varied: m; everything else fixed):
  dataset SIFT-1M, nlist = 4096 (one shared, seeded k-means), nbits = 8, m in {8, 16, 32, 64}
  variant "IVF-PQ"          : plain search, top 10 by PQ (approximate) distance
  variant "IVF-PQ+refine"   : IndexRefineFlat, k_factor = 10 -> PQ top 100 re-ranked exactly
  Both variants share the SAME built IVF-PQ index, so for a given nprobe the re-ranked result
  is computed from exactly the candidate pool that plain search with k = 100 returns.

Protocol (as Part 2): tune sweep over nprobe on the first 5,000 queries (recall on all 5,000;
QPS single-query, 1 thread, pinned, on 1,000; R = 3); select per (variant, m) the tune Pareto
front plus the fastest configuration reaching each target recall; eval only those on the last
5,000 queries (recall on 5,000; QPS on 1,000, R = 5). Eval rows of the re-ranked variant also
time the candidate generation alone (plain search with k = 100), so the re-ranking cost per
query = 1/QPS(refine) - 1/QPS(candidates).

Usage (from code/):  python -m experiments.part3_compression [--quick] [--dev]
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time

import numpy as np

from src import threads
from src.config import K, METADATA_DIR, RESULTS_DIR, SEED
from src.data_loader import load_dataset
from src.indexes import BuildSpec, CoarseQuantizerCache, add_exact_refine, build, index_size_bytes
from src.pareto import pareto_front, select_for_target
from src.provenance import HypothesisGuardError, require_committed_hypothesis, run_metadata
from src.recall import pool_recall, recall_at_k, true_sq_distances
from src.results_io import replace_rows
from src.timing import benchmark_lock, time_search
from experiments.part2_ann import load_reference_tau, exact_qps, STOP_RECALL, PLATEAU_GAIN

EXPERIMENT = "p3_compression"
K_FACTOR = 10
POOL = K * K_FACTOR                                   # 100 candidates
TARGETS = (0.50, 0.80, 0.90, 0.95)
NPROBE = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
GRID = {"full": {"nlist": 4096, "m": (8, 16, 32, 64), "nprobe": NPROBE},
        "quick": {"nlist": 256, "m": (8, 16, 32), "nprobe": [1, 2, 4, 8, 16, 32, 64, 128]}}
KEY = ("experiment_id", "dataset_id", "n_base", "index_type", "build_params")


def measure(ds, plain, index, queries, tau, n_time, repeats, split, time_candidates: bool) -> dict:
    """Recall of `index` on all queries, pool recall of the plain top-100, single-query QPS."""
    threads.build_phase()                                       # untimed accuracy measurements
    ids = index.search(queries, K)[1]
    rec = recall_at_k(ids, true_sq_distances(queries, ds.train, ids), tau)
    pool = plain.search(queries, POOL)[1]
    prec = pool_recall(pool, true_sq_distances(queries, ds.train, pool), tau)
    st = threads.search_phase()                                 # timed, 1 thread, pinned
    n = min(n_time, len(queries))
    t = time_search(index, queries[:n], K, repeats, "single")
    row = {"query_split": split, "n_queries": len(queries), "recall_at_10": round(float(rec.mean()), 6),
           "pool_recall_10at100": round(float(prec.mean()), 6), "thread_check": t.thread_check(True),
           "pinned": st.pinned, "pin_fallback_reason": st.fallback_reason}
    row.update(t.as_row())
    row.update(st.as_row("search"))
    match = bool(np.array_equal(t.ids, ids[:n]))
    if time_candidates:
        c = time_search(plain, queries[:n], POOL, repeats, "single")
        row["candidate_qps_median"] = round(c.qps_median, 3)
        row["rerank_ms_per_query"] = round(1000.0 / t.qps_median - 1000.0 / c.qps_median, 4)
    return row, match, float(prec.mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dev", action="store_true", help="allow uncommitted source changes; rows are not reportable")
    ap.add_argument("--n-time-tune", type=int, default=1000)
    ap.add_argument("--repeats-tune", type=int, default=3)
    ap.add_argument("--n-time-eval", type=int, default=1000)
    ap.add_argument("--repeats-eval", type=int, default=5)
    args = ap.parse_args(argv)

    try:                                                         # decision A6: before ANY work
        hyp = require_committed_hypothesis("Part 3", require_clean_tree=not args.dev)
    except HypothesisGuardError as e:
        print(e, file=sys.stderr)
        return e.exit_code
    print(f"hypothesis guard OK: commit {hyp['hypothesis_commit'][:7]}, section sha256 {hyp['hypothesis_sha256'][:16]}")

    grid = GRID["quick" if args.quick else "full"]
    results_dir = RESULTS_DIR / "quick" if args.quick else RESULTS_DIR
    metadata_dir = results_dir / "metadata" if args.quick else METADATA_DIR
    metadata_dir.mkdir(parents=True, exist_ok=True)
    provenance = {"hypothesis_commit": hyp["hypothesis_commit"], "hypothesis_sha256": hyp["hypothesis_sha256"]}

    with benchmark_lock(RESULTS_DIR, "part3_compression"):
        t_start = time.perf_counter()
        ds = load_dataset("sift", max_base=100_000 if args.quick else None)
        tau = load_reference_tau(ds)
        ex = exact_qps(ds.dataset_id, results_dir / "results.csv")
        n = len(ds.train)
        tune_q, tau_tune = ds.queries[ds.split.tune], tau[ds.split.tune]
        eval_q, tau_eval = ds.queries[ds.split.eval], tau[ds.split.eval]
        coarse = CoarseQuantizerCache(ds.train)
        build_log = {}
        print(f"[{ds.dataset_id}] n={n} nlist={grid['nlist']} m={grid['m']} exact single QPS={ex}")

        for m in grid["m"]:
            meta = dict(run_metadata(EXPERIMENT), **provenance)
            spec = BuildSpec.make("IVF-PQ", nlist=grid["nlist"], m=m, nbits=8)
            bst = threads.build_phase()
            res = build(spec, ds.train, coarse)
            refine, copy_s = add_exact_refine(res.index, ds.train, K_FACTOR)
            sizes = {"IVF-PQ": index_size_bytes(res.index), "IVF-PQ+refine": index_size_bytes(refine)}
            build_log[spec.name] = {"build_time_s": res.build_time_s, **res.timings, "refine_copy_s": copy_s,
                                    "index_size_bytes": sizes, "build_threads": bst.as_row("build")}
            print(f" m={m}: built in {res.build_time_s:.1f} s (+{copy_s:.2f} s raw-vector copy for refine); "
                  f"{sizes['IVF-PQ'] / n:.1f} B/vec plain, {sizes['IVF-PQ+refine'] / n:.1f} B/vec with refine")
            variants = {"IVF-PQ": (res.index, res.build_time_s, dict(spec.p)),
                        "IVF-PQ+refine": (refine, res.build_time_s + copy_s,
                                          dict(spec.p, refine="Flat", k_factor=K_FACTOR))}
            rows = []
            for vname, (index, btime, bparams) in variants.items():
                base = dict(meta, dataset_id=ds.dataset_id, n_base=n, index_type=vname, build_params=bparams,
                            build_time_s=round(btime, 3), index_size_bytes=sizes[vname],
                            bytes_per_vector=round(sizes[vname] / n, 2), seed=SEED, status="ok",
                            search_mode="single", **bst.as_row("build"))
                tune_rows, history = [], []
                for v in grid["nprobe"]:
                    if v > grid["nlist"]:
                        break
                    res.index.nprobe = v
                    m_row, match, _ = measure(ds, res.index, index, tune_q, tau_tune, args.n_time_tune,
                                              args.repeats_tune, "tune", time_candidates=False)
                    row = dict(base, **m_row, search_params={"nprobe": v} | ({"k_factor": K_FACTOR} if vname != "IVF-PQ" else {}),
                               notes=f"tuning sweep; single==batch ids on timed subset: {match}")
                    tune_rows.append(row)
                    history.append(row["recall_at_10"])
                    print(f"   tune {vname:14s} nprobe={v:5d} recall {row['recall_at_10']:.4f} "
                          f"pool10@100 {row['pool_recall_10at100']:.4f} qps {row['qps_median']:9.1f}"
                          f"{'' if match else '  IDS MISMATCH'}")
                    if row["recall_at_10"] >= STOP_RECALL:
                        break
                    if ex and row["qps_median"] < ex:
                        print("   stop: QPS below exact search")
                        break
                    if len(history) >= 3 and history[-1] - history[-2] < PLATEAU_GAIN and history[-2] - history[-3] < PLATEAU_GAIN:
                        print("   stop: recall plateau")
                        break

                pts = [{"recall": r["recall_at_10"], "qps": r["qps_median"], "row": r} for r in tune_rows]
                chosen = {id(p["row"]): ["pareto"] for p in pareto_front(pts)}
                for t in TARGETS:
                    sel = select_for_target(pts, t)
                    if sel is not None:
                        chosen.setdefault(id(sel["row"]), ["pareto"]).append(f"target{t:.2f}")
                eval_meta = dict(run_metadata(EXPERIMENT), **provenance)
                for r in tune_rows:
                    if id(r) not in chosen:
                        continue
                    r["notes"] += "; selected for eval: " + ",".join(chosen[id(r)])
                    res.index.nprobe = r["search_params"]["nprobe"]
                    m_row, match, _ = measure(ds, res.index, index, eval_q, tau_eval, args.n_time_eval,
                                              args.repeats_eval, "eval", time_candidates=(vname != "IVF-PQ"))
                    keep = {k: r[k] for k in ("dataset_id", "n_base", "index_type", "build_params", "build_time_s",
                                              "index_size_bytes", "bytes_per_vector", "seed", "status",
                                              "build_affinity", "build_omp", "build_blas", "search_params")}
                    er = dict(eval_meta, **keep, **m_row,
                              notes=f"eval of tune-selected config ({','.join(chosen[id(r)])}); "
                                    f"single==batch ids on timed subset: {match}")
                    rows.append(er)
                    extra = (f" cand qps {er['candidate_qps_median']:.1f} rerank {er['rerank_ms_per_query']:.3f} ms"
                             if "candidate_qps_median" in er else "")
                    print(f"   EVAL {vname:14s} nprobe={r['search_params']['nprobe']:5d} recall {er['recall_at_10']:.4f} "
                          f"pool {er['pool_recall_10at100']:.4f} qps {er['qps_median']:9.1f}{extra} "
                          f"[{','.join(chosen[id(r)])}]{'' if match else '  IDS MISMATCH'}")
                rows.extend(tune_rows)
            replace_rows(rows, results_dir / "results.csv", key=KEY)
            (metadata_dir / f"part3_{ds.dataset_id}_builds.json").write_text(json.dumps(build_log, indent=2))
            del refine, res, variants
            gc.collect()
        print(f"[{ds.dataset_id}] Part 3 done in {(time.perf_counter() - t_start) / 60:.1f} min")
    threads.search_phase()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
