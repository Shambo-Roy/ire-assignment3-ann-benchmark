"""Diagnostic (not a benchmark): single-query exact-search QPS on SIFT pinned to each fast
logical CPU, two E-core CPUs and unpinned, in two rounds of opposite order to expose drift.

Written after the M2 calibration showed unpinned (37.7 QPS) faster than pinned to logical 10
(35.8). Output: results/metadata/pinning_probe.json. Run from code/:
    .venv/Scripts/python -m experiments.probes.pinning_probe
"""

from __future__ import annotations

import json
import statistics

import faiss

from src.config import METADATA_DIR
from src.data_loader import load_dataset
from src import threads
from src.provenance import run_metadata
from src.timing import benchmark_lock, time_search

N_Q, REPEATS = 100, 2


def main() -> None:
    meta = run_metadata("probe_pinning")
    ds = load_dataset("sift")
    threads.build_phase()
    index = faiss.IndexFlatL2(ds.train.shape[1])
    index.add(ds.train)
    q = ds.eval_queries[:N_Q]
    topo, _ = threads.cpu_topology()
    fast = [c.logical for c in threads.fast_cpus(topo)]
    slow = [c.logical for c in topo if c.logical not in fast][:2]
    modes = [(f"cpu{c}", (lambda c=c: threads.search_phase(c))) for c in fast + slow]
    modes.append(("unpinned", threads.unpinned_single_thread))
    out = {"meta": meta, "n_queries": N_Q, "repeats": REPEATS, "fast_cpus": fast, "rounds": []}
    for rnd, order in enumerate((modes, modes[::-1])):
        res = {}
        for label, phase in order:
            phase()
            t = time_search(index, q, 10, REPEATS, "single")
            res[label] = {"qps_median": round(t.qps_median, 2), "qps": [round(x, 2) for x in t.qps],
                          "cpu_wall": round(t.cpu_wall_median, 3)}
            print(f"round {rnd} {label:9s} {t.qps_median:6.2f} qps")
        out["rounds"].append(res)
    out["mean_of_round_medians"] = {lab: round(statistics.mean(r[lab]["qps_median"] for r in out["rounds"]), 2)
                                    for lab, _ in modes}
    threads.search_phase()
    (METADATA_DIR / "pinning_probe.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["mean_of_round_medians"]))


if __name__ == "__main__":
    with benchmark_lock(label="pinning_probe"):
        main()
