"""Part 4: does increasing dimensionality make ANN search harder? (seeded synthetic data)

Tests the author's hypotheses 4.1 and 4.2 (hypothesis.md, committed in 55016fe). The hypothesis
guard runs FIRST and the script refuses to start (exit 3) unless the Part 4 section is committed and
unchanged. Rows record hypothesis_commit and hypothesis_sha256.

One property varied: the dimensionality d in {32, 64, 128 (control), 256, 512}.
Held fixed for every d: generator (Gaussian -> L2-normalised, src/synthetic.py), seed policy,
n_base = 100,000, n_query = 1,000, Euclidean metric, build settings (HNSW M=16 efC=200;
IVF-Flat nlist=1024, seeded k-means), the pre-specified search settings below, the tie-aware
recall@10, and the timing method (single query, 1 thread, pinned P-core, median of R).

The search settings are fixed in advance and NOT tuned on the data, so there is no tuning step and
all 1,000 queries are used for every measurement (query_split = "all").

Usage (from code/):  python -m experiments.part4_dimensionality [--quick] [--dev]
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time

import faiss
import numpy as np

from src import threads
from src.config import K, METADATA_DIR, RESULTS_DIR, SEED
from src.indexes import BuildSpec, build, index_size_bytes, set_search_param
from src.provenance import HypothesisGuardError, require_committed_hypothesis, run_metadata
from src.recall import boundary_tie_stats, recall_at_k, reference_from_candidates, true_sq_distances
from src.results_io import replace_rows
from src.synthetic import distance_stats, generate
from src.timing import benchmark_lock, time_search
from experiments.part1_exact import blas_threshold

EXPERIMENT = "p4_dim"
REF_WIDTH = 100
SETTINGS = {
    "full": {"dims": (32, 64, 128, 256, 512), "n_base": 100_000, "n_query": 1_000,
             "hnsw": {"M": 16, "efC": 200}, "ef": (16, 32, 64, 128, 256),
             "ivf": {"nlist": 1024}, "nprobe": (1, 4, 16, 64, 256)},
    "quick": {"dims": (32, 128, 512), "n_base": 20_000, "n_query": 200,
              "hnsw": {"M": 16, "efC": 200}, "ef": (16, 64, 256),
              "ivf": {"nlist": 256}, "nprobe": (1, 16, 64)},
}


def recall_of(ids, q, base, tau):
    return float(recall_at_k(ids, true_sq_distances(q, base, ids), tau).mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dev", action="store_true", help="allow uncommitted source changes; rows are not reportable")
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args(argv)

    try:                                                         # decision A6: before ANY work
        hyp = require_committed_hypothesis("Part 4", require_clean_tree=not args.dev)
    except HypothesisGuardError as e:
        print(e, file=sys.stderr)
        return e.exit_code
    print(f"hypothesis guard OK: commit {hyp['hypothesis_commit'][:7]}, section sha256 {hyp['hypothesis_sha256'][:16]}")

    cfg = SETTINGS["quick" if args.quick else "full"]
    results_dir = RESULTS_DIR / "quick" if args.quick else RESULTS_DIR
    metadata_dir = results_dir / "metadata" if args.quick else METADATA_DIR
    metadata_dir.mkdir(parents=True, exist_ok=True)
    provenance = {"hypothesis_commit": hyp["hypothesis_commit"], "hypothesis_sha256": hyp["hypothesis_sha256"]}
    stats_path = metadata_dir / "part4_distance_stats.json"
    all_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}

    with benchmark_lock(RESULTS_DIR, "part4_dimensionality"):
        t_start = time.perf_counter()
        for d in cfg["dims"]:
            meta = dict(run_metadata(EXPERIMENT), **provenance)
            t0 = time.perf_counter()
            ds = generate(d, cfg["n_base"], cfg["n_query"], SEED)
            gen_s = time.perf_counter() - t0
            x, q = ds.base, ds.queries
            n = len(x)

            # exact reference (untimed): Flat top-100 re-scored in float64
            bst = threads.build_phase()
            flat = faiss.IndexFlatL2(d)
            t0 = time.perf_counter()
            flat.add(x)
            flat_build = time.perf_counter() - t0
            with blas_threshold(20):                       # fast path; correctness from float64 re-scoring
                _, cand = flat.search(q, REF_WIDTH)
            ref_ids, ref_d = reference_from_candidates(q, x, cand)
            tau = ref_d[:, K - 1]
            st = distance_stats(q, x, ref_d, K)
            st.update({"fingerprint_sha256": ds.fingerprint(), "generate_s": round(gen_s, 2),
                       "boundary_ties": boundary_tie_stats(ref_d, K), "dataset_id": ds.dataset_id,
                       "seed_policy": f"SeedSequence([{SEED}, {d}]).spawn(2) -> (base, queries)", **meta})
            all_stats[str(d)] = st
            print(f"[d={d}] {ds.dataset_id}: mean dist {st['mean_distance']:.4f} std {st['std_distance']:.4f} "
                  f"(cv {st['cv_distance']:.4f}) RC {st['median_relative_contrast']:.3f} "
                  f"gap10/11 {st['median_relative_gap_10_11']:.5f}")

            base_row = dict(meta, dataset_id=ds.dataset_id, n_base=n, query_split="all", n_queries=len(q),
                            seed=f"{SEED}/d{d}", status="ok", **bst.as_row("build"))
            rows = []

            def timed(index, index_type, bparams, sparams, btime, size, note):
                sst = threads.search_phase()
                t = time_search(index, q, K, args.repeats, "single")
                threads.build_phase()
                ids_b = index.search(q, K)[1]
                rec = recall_of(t.ids, q, x, tau)
                r = dict(base_row, index_type=index_type, build_params=bparams, search_params=sparams,
                         build_time_s=round(btime, 3), index_size_bytes=size, bytes_per_vector=round(size / n, 2),
                         recall_at_10=round(rec, 6), thread_check=t.thread_check(True), pinned=sst.pinned,
                         pin_fallback_reason=sst.fallback_reason,
                         notes=f"{note}; single==batch ids: {bool(np.array_equal(t.ids, ids_b))}")
                r.update(t.as_row())
                r.update(sst.as_row("search"))
                rows.append(r)
                print(f"   {index_type:8s} {json.dumps(sparams):20s} recall {rec:.4f} qps {t.qps_median:9.1f}")

            timed(flat, "Flat", {}, {}, flat_build, index_size_bytes(flat), "exact baseline")
            del flat
            gc.collect()

            for itype, bp, sweep in (("HNSW", cfg["hnsw"], cfg["ef"]), ("IVF-Flat", cfg["ivf"], cfg["nprobe"])):
                spec = BuildSpec.make(itype, **bp)
                threads.build_phase()
                res = build(spec, x)                             # fresh seeded k-means per d for IVF
                size = index_size_bytes(res.index)
                print(f"   built {spec.name}: {res.build_time_s:.1f} s, {size / n:.1f} B/vec")
                for v in sweep:
                    if itype == "IVF-Flat" and v > bp["nlist"]:
                        continue
                    set_search_param(res.index, spec, v)
                    timed(res.index, itype, spec.p, {spec.search_param: v}, res.build_time_s, size,
                          "fixed pre-specified search setting")
                del res
                gc.collect()

            replace_rows(rows, results_dir / "results.csv")
            stats_path.write_text(json.dumps(all_stats, indent=2))
            del ds, x, q
            gc.collect()
        print(f"Part 4 done in {(time.perf_counter() - t_start) / 60:.1f} min")
    threads.search_phase()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
