"""Part 2 figures and tables, generated only from results.csv (no new measurements).

Outputs (under results/ or results/quick/):
  figures/part2_<dataset>_recall_vs_qps.png   eval measurements of each index's tune-selected Pareto
                                              configurations, plus the exact single-query baseline
  tables/part2_qps_at_recall.{csv,md}         QPS at recall 0.90 / 0.95 (config chosen on tune, numbers on eval)
  tables/part2_build_cost_at_095.{csv,md}     build time, index size, bytes/vector of the 0.95 configuration
  tables/part2_max_recall.{csv,md}            highest tune and eval recall reached per index and build

Usage (from code/):  python -m experiments.part2_report [--quick]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import RESULTS_DIR
from src.results_io import read_rows

INDEX_ORDER = ("HNSW", "IVF-Flat", "IVF-PQ")
COLORS = {"HNSW": "#1f77b4", "IVF-Flat": "#2ca02c", "IVF-PQ": "#d62728", "Exact": "#000000"}
MARKERS = {"HNSW": "o", "IVF-Flat": "s", "IVF-PQ": "^"}
TARGETS = ("0.90", "0.95")


def _f(r, k):
    return float(r[k]) if r.get(k) not in (None, "") else None


def label(r) -> str:
    bp, sp = json.loads(r["build_params"]), json.loads(r["search_params"])
    keep = {k: v for k, v in bp.items() if k not in ("efC", "nbits")}
    return ",".join(f"{k}={v}" for k, v in keep.items()) + " " + ",".join(f"{k}={v}" for k, v in sp.items())


def write_table(rows: list[dict], cols: list[str], path_base, title: str) -> None:
    with open(f"{path_base}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    lines = [f"# {title}", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    with open(f"{path_base}.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    out = RESULTS_DIR / "quick" if args.quick else RESULTS_DIR
    rows = read_rows(out / "results.csv")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    p2 = [r for r in rows if r["experiment_id"] == "p2_ann" and r["status"] == "ok"]
    datasets = sorted({r["dataset_id"] for r in p2}, key=lambda d: (not d.startswith("sift"), d))
    qps_tab, cost_tab, max_tab = [], [], []

    for ds in datasets:
        ev = [r for r in p2 if r["dataset_id"] == ds and r["query_split"] == "eval"]
        tu = [r for r in p2 if r["dataset_id"] == ds and r["query_split"] == "tune"]
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for it in INDEX_ORDER:
            pts = sorted((r for r in ev if r["index_type"] == it), key=lambda r: _f(r, "recall_at_10"))
            if pts:
                ax.plot([_f(r, "recall_at_10") for r in pts], [_f(r, "qps_median") for r in pts],
                        marker=MARKERS[it], ms=4, lw=1.4, color=COLORS[it], label=it)
            for t in TARGETS:
                sel = [r for r in ev if r["index_type"] == it and f"target{t}" in r["notes"]]
                if sel:
                    r = sel[0]
                    qps_tab.append({"dataset": ds, "index": it, "target_recall": t, "config": label(r),
                                    "eval_recall": f"{_f(r, 'recall_at_10'):.4f}",
                                    "eval_qps_median": f"{_f(r, 'qps_median'):.1f}",
                                    "meets_target_on_eval": _f(r, "recall_at_10") >= float(t)})
                    if t == "0.95":
                        cost_tab.append({"dataset": ds, "index": it, "config": label(r),
                                         "build_time_s": f"{_f(r, 'build_time_s'):.1f}",
                                         "index_size_MB": f"{_f(r, 'index_size_bytes') / 1e6:.1f}",
                                         "bytes_per_vector": f"{_f(r, 'bytes_per_vector'):.1f}",
                                         "eval_recall": f"{_f(r, 'recall_at_10'):.4f}"})
                else:
                    best = max((_f(r, "recall_at_10") for r in tu if r["index_type"] == it), default=None)
                    qps_tab.append({"dataset": ds, "index": it, "target_recall": t,
                                    "config": "not reached on tune",
                                    "eval_recall": "", "eval_qps_median": "",
                                    "meets_target_on_eval": f"max tune recall {best:.4f}" if best is not None else ""})
                    if t == "0.95":
                        cost_tab.append({"dataset": ds, "index": it, "config": "recall 0.95 not reached",
                                         "build_time_s": "", "index_size_MB": "", "bytes_per_vector": "",
                                         "eval_recall": f"max tune recall {best:.4f}" if best is not None else ""})
        builds = defaultdict(list)
        for r in tu:
            builds[(r["index_type"], r["build_params"])].append(r)
        for (it, bp), rs in sorted(builds.items()):
            b = max(rs, key=lambda r: _f(r, "recall_at_10"))
            e = [r for r in ev if r["index_type"] == it and r["build_params"] == bp]
            max_tab.append({"dataset": ds, "index": it,
                            "build": ",".join(f"{k}={v}" for k, v in json.loads(bp).items()),
                            "max_tune_recall": f"{_f(b, 'recall_at_10'):.4f}", "at": label(b).split(" ", 1)[1],
                            "max_eval_recall": f"{max(_f(r, 'recall_at_10') for r in e):.4f}" if e else "",
                            "build_time_s": f"{_f(b, 'build_time_s'):.1f}",
                            "bytes_per_vector": f"{_f(b, 'bytes_per_vector'):.1f}"})

        exact = [r for r in rows if r["experiment_id"] == "p1_exact" and r["dataset_id"] == ds
                 and r["notes"].startswith("primary: one query")]
        if exact:
            ax.scatter([_f(exact[0], "recall_at_10")], [_f(exact[0], "qps_median")], marker="*", s=120,
                       color=COLORS["Exact"], zorder=5, label="Exact (Flat)")
        for t in TARGETS:
            ax.axvline(float(t), color="0.75", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_xlabel("recall@10 (eval queries)")
        ax.set_ylabel("queries per second (1 thread, log scale)")
        ax.set_title(f"{ds}: recall@10 vs QPS
(Pareto configs chosen on tune queries, measured on eval queries)", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "figures" / f"part2_{ds}_recall_vs_qps.png", dpi=160)
        plt.close(fig)
        print(f"figure: {out / 'figures' / f'part2_{ds}_recall_vs_qps.png'}")

    write_table(qps_tab, ["dataset", "index", "target_recall", "config", "eval_recall", "eval_qps_median",
                          "meets_target_on_eval"], out / "tables" / "part2_qps_at_recall",
                "QPS at recall 0.90 / 0.95 (configuration chosen on tune queries, measured on eval queries)")
    write_table(cost_tab, ["dataset", "index", "config", "build_time_s", "index_size_MB", "bytes_per_vector",
                           "eval_recall"], out / "tables" / "part2_build_cost_at_095",
                "Build time, index size and bytes per vector of the recall-0.95 configuration")
    write_table(max_tab, ["dataset", "index", "build", "max_tune_recall", "at", "max_eval_recall",
                          "build_time_s", "bytes_per_vector"], out / "tables" / "part2_max_recall",
                "Highest recall reached per index and build")
    print(f"tables: {out / 'tables'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
