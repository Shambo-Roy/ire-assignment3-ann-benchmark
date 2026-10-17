"""Build-scaling check (decision A5): how much of each build time comes from multi-threading.

For one configuration per index type on SIFT, build once under build_phase() (all P-core logical
CPUs, OpenMP = OpenBLAS = their count) and once single-threaded (one pinned P-core CPU,
OpenMP = OpenBLAS = 1). Each build runs its own k-means so it is timed under its own threads.
Reported, not used to choose between indexes. Rows: experiment_id = p2_build_scaling.

Usage (from code/):  python -m experiments.build_scaling [--quick]
"""

from __future__ import annotations

import argparse
import gc

from src import threads
from src.config import RESULTS_DIR, SEED
from src.data_loader import load_dataset
from src.indexes import BuildSpec, CoarseQuantizerCache, build, index_size_bytes
from src.provenance import run_metadata
from src.results_io import replace_rows
from src.timing import benchmark_lock

FULL = [BuildSpec.make("HNSW", M=16, efC=200), BuildSpec.make("IVF-Flat", nlist=1024),
        BuildSpec.make("IVF-PQ", nlist=1024, m=16, nbits=8)]
QUICK = [BuildSpec.make("HNSW", M=16, efC=200), BuildSpec.make("IVF-Flat", nlist=256),
         BuildSpec.make("IVF-PQ", nlist=256, m=16, nbits=8)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    results_dir = RESULTS_DIR / "quick" if args.quick else RESULTS_DIR
    meta = run_metadata("p2_build_scaling")
    ds = load_dataset("sift", max_base=100_000 if args.quick else None)
    n = len(ds.train)
    rows = []
    with benchmark_lock(RESULTS_DIR, "build_scaling"):
        for spec in (QUICK if args.quick else FULL):
            times = {}
            for label, phase in (("P-core threads", threads.build_phase), ("1 thread", threads.search_phase)):
                st = phase()
                res = build(spec, ds.train, CoarseQuantizerCache(ds.train))
                times[label] = res.build_time_s
                rows.append(dict(meta, dataset_id=ds.dataset_id, n_base=n, index_type=spec.index_type,
                                 build_params=spec.p, build_time_s=round(res.build_time_s, 3),
                                 index_size_bytes=index_size_bytes(res.index),
                                 seed=SEED if spec.index_type != "HNSW" else "faiss-default",
                                 build_affinity=" ".join(map(str, st.affinity)), build_omp=st.omp_threads,
                                 build_blas=st.blas_threads, status="ok", query_split="n/a",
                                 notes=f"build scaling: {label}; breakdown "
                                       + ", ".join(f"{k}={v:.2f}s" for k, v in res.timings.items())))
                print(f"  {spec.name:32s} {label:15s} {res.build_time_s:8.1f} s  "
                      + ", ".join(f"{k}={v:.1f}" for k, v in res.timings.items()), flush=True)
                del res
                gc.collect()
            print(f"  -> speed-up {times['1 thread'] / times['P-core threads']:.2f}x")
            for r in rows[-2:]:
                r["notes"] += f"; speed-up 1T/P-core = {times['1 thread'] / times['P-core threads']:.2f}x"
    threads.search_phase()
    replace_rows(rows, results_dir / "results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
