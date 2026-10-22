"""Part 3 figure and tables, generated only from results.csv (rows p3_compression).

Outputs (under results/ or results/quick/):
  figures/part3_compression.png      (a) eval recall@10 vs QPS per m, plain IVF-PQ vs + exact re-ranking;
                                     (b) best eval recall vs bytes per vector (memory), with the
                                         candidate-pool recall 10@100 that bounds re-ranking
  tables/part3_by_m.{csv,md}         per m: memory, build time, best recall plain / re-ranked / pool,
                                     QPS at recall targets for both variants (chosen on tune, eval numbers)
  tables/part3_rerank_cost.{csv,md}  per evaluated re-ranked config: candidate QPS, refine QPS, ms per query

Usage (from code/):  python -m experiments.part3_report [--quick]
"""

from __future__ import annotations

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from src.config import RESULTS_DIR
from src.results_io import read_rows
from experiments.part2_report import write_table, _f

TARGETS = ("0.50", "0.80", "0.90", "0.95")
VARIANTS = ("IVF-PQ", "IVF-PQ+refine")
CMAP = plt.get_cmap("viridis")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    out = RESULTS_DIR / "quick" if args.quick else RESULTS_DIR
    rows = [r for r in read_rows(out / "results.csv") if r["experiment_id"] == "p3_compression" and r["status"] == "ok"]
    if not rows:
        print("no p3_compression rows")
        return 1
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    ev = [r for r in rows if r["query_split"] == "eval"]
    tu = [r for r in rows if r["query_split"] == "tune"]
    ms = sorted({json.loads(r["build_params"])["m"] for r in rows})
    color = {m: CMAP(i / max(1, len(ms) - 1) * 0.85) for i, m in enumerate(ms)}
    m_of = lambda r: json.loads(r["build_params"])["m"]

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.35, 1]})
    by_m, cost = [], []
    for m in ms:
        rec = {}
        for v in VARIANTS:
            pts = sorted((r for r in ev if m_of(r) == m and r["index_type"] == v), key=lambda r: _f(r, "recall_at_10"))
            if pts:
                ax.plot([_f(r, "recall_at_10") for r in pts], [_f(r, "qps_median") for r in pts],
                        ls="-" if v != "IVF-PQ" else "--", marker="o" if v != "IVF-PQ" else "x", ms=4,
                        color=color[m], label=f"m={m} {'+ re-rank 100' if v != 'IVF-PQ' else 'plain'}")
            best = max(pts, key=lambda r: _f(r, "recall_at_10")) if pts else None
            best_tune = max((r for r in tu if m_of(r) == m and r["index_type"] == v), key=lambda r: _f(r, "recall_at_10"), default=None)
            rec[v] = (best, best_tune)
            for r in (r for r in ev if m_of(r) == m and r["index_type"] == v and r["candidate_qps_median"]):
                cost.append({"m": m, "nprobe": json.loads(r["search_params"])["nprobe"],
                             "eval_recall_refine": f"{_f(r, 'recall_at_10'):.4f}",
                             "pool_recall_10at100": f"{_f(r, 'pool_recall_10at100'):.4f}",
                             "refine_qps": f"{_f(r, 'qps_median'):.1f}",
                             "candidate_qps_k100": f"{_f(r, 'candidate_qps_median'):.1f}",
                             "rerank_ms_per_query": f"{_f(r, 'rerank_ms_per_query'):.4f}",
                             "rerank_share_of_latency": f"{_f(r, 'rerank_ms_per_query') * _f(r, 'qps_median') / 1000:.1%}"})
        plain, ref = rec["IVF-PQ"][0], rec["IVF-PQ+refine"][0]
        row = {"m": m, "code_bytes": m,
               "bytes_per_vector_plain": f"{_f(plain, 'bytes_per_vector'):.1f}" if plain else "",
               "bytes_per_vector_refine": f"{_f(ref, 'bytes_per_vector'):.1f}" if ref else "",
               "build_time_s": f"{_f(plain, 'build_time_s'):.1f}" if plain else "",
               "best_eval_recall_plain": f"{_f(plain, 'recall_at_10'):.4f}" if plain else "",
               "best_eval_recall_refine": f"{_f(ref, 'recall_at_10'):.4f}" if ref else "",
               "pool_recall_at_that_nprobe": f"{_f(ref, 'pool_recall_10at100'):.4f}" if ref else "",
               "best_tune_recall_plain": f"{_f(rec['IVF-PQ'][1], 'recall_at_10'):.4f}" if rec["IVF-PQ"][1] else "",
               "best_tune_recall_refine": f"{_f(rec['IVF-PQ+refine'][1], 'recall_at_10'):.4f}" if rec["IVF-PQ+refine"][1] else ""}
        for t in TARGETS:
            for v, short in (("IVF-PQ", "plain"), ("IVF-PQ+refine", "refine")):
                sel = [r for r in ev if m_of(r) == m and r["index_type"] == v and f"target{t}" in r["notes"]]
                row[f"qps@{t}_{short}"] = (f"{_f(sel[0], 'qps_median'):.0f} (r={_f(sel[0], 'recall_at_10'):.3f}, "
                                           f"nprobe={json.loads(sel[0]['search_params'])['nprobe']})") if sel else "not reached"
        by_m.append(row)
        if plain and ref:
            bpv_p, bpv_r = _f(plain, "bytes_per_vector"), _f(ref, "bytes_per_vector")
            bx.scatter([bpv_p], [_f(plain, "recall_at_10")], marker="x", s=50, color=color[m])
            bx.scatter([bpv_r], [_f(ref, "recall_at_10")], marker="o", s=40, color=color[m])
            bx.scatter([bpv_r], [_f(ref, "pool_recall_10at100")], marker="_", s=200, color=color[m])
            bx.annotate(f"m={m}", (bpv_p, _f(plain, "recall_at_10")), textcoords="offset points", xytext=(5, -10), fontsize=8)

    ax.set_yscale("log")
    ax.set_xlabel("recall@10 (eval queries)")
    ax.set_ylabel("QPS (1 thread, log scale)")
    nlist = json.loads(rows[0]["build_params"])["nlist"]
    ax.set_title(f"(a) {rows[0]['dataset_id']}, IVF-PQ nlist={nlist}: plain (dashed) vs exact re-rank of top 100 (solid)", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    bx.set_xscale("log")
    ticks = [8, 16, 32, 64, 128, 256, 512, 1024]
    bx.set_xticks(ticks)
    bx.set_xticklabels([str(t) for t in ticks])
    bx.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    bx.set_xlabel("bytes per vector (log scale)")
    bx.set_ylabel("best eval recall@10")
    bx.set_title("(b) memory vs best recall: x plain, o re-ranked, — pool recall 10@100", fontsize=9)
    bx.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "figures" / "part3_compression.png", dpi=160)
    plt.close(fig)
    print(f"figure: {out / 'figures' / 'part3_compression.png'}")

    cols = list(by_m[0].keys())
    write_table(by_m, cols, out / "tables" / "part3_by_m",
                "Part 3: IVF-PQ by number of sub-vectors m (config chosen on tune queries; numbers from eval queries)")
    if cost:
        write_table(cost, list(cost[0].keys()), out / "tables" / "part3_rerank_cost",
                    "Part 3: cost of exact re-ranking of the top 100 (eval; rerank ms = 1/QPS(refine) - 1/QPS(PQ, k=100))")
    print(f"tables: {out / 'tables'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
