"""Part 4 main figure and tables, generated only from results.csv (rows p4_dim) and the
distance statistics in results/metadata/part4_distance_stats.json.

Outputs (under results/ or results/quick/):
  figures/part4_dimensionality.png     (a) recall@10 vs d at each fixed search setting (HNSW, IVF-Flat)
                                       (b) single-thread QPS vs d (incl. exact search)
                                       (c) distance concentration vs d: relative contrast, CV, 10/11 gap
  tables/part4_recall_by_d.{csv,md}    recall and change vs the d = 128 control, per index and setting
  tables/part4_qps_by_d.{csv,md}       QPS per index and setting
  tables/part4_distance_stats.{csv,md} distance statistics per d

Usage (from code/):  python -m experiments.part4_report [--quick]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import METADATA_DIR, RESULTS_DIR
from src.results_io import read_rows
from experiments.part2_report import write_table, _f

CONTROL = 128


def setting_label(r) -> str:
    sp = json.loads(r["search_params"])
    return r["index_type"] + (" " + ",".join(f"{k}={v}" for k, v in sp.items()) if sp else "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    out = RESULTS_DIR / "quick" if args.quick else RESULTS_DIR
    meta_dir = out / "metadata" if args.quick else METADATA_DIR
    rows = [r for r in read_rows(out / "results.csv") if r["experiment_id"] == "p4_dim" and r["status"] == "ok"]
    stats = json.loads((meta_dir / "part4_distance_stats.json").read_text())
    if not rows:
        print("no p4_dim rows")
        return 1
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)

    d_of = lambda r: int(r["dataset_id"].split("_d")[1].split("_")[0])
    dims = sorted({d_of(r) for r in rows})
    series = defaultdict(dict)                          # label -> d -> row
    for r in rows:
        series[setting_label(r)][d_of(r)] = r

    def style(label):
        if label.startswith("HNSW"):
            v = int(label.split("=")[1]); return "o", "-", plt.get_cmap("Blues")(0.35 + 0.6 * (v.bit_length() - 4) / 5)
        if label.startswith("IVF"):
            v = int(label.split("=")[1]); return "s", "--", plt.get_cmap("Greens")(0.35 + 0.6 * v.bit_length() / 9)
        return "*", "-", "black"

    order = sorted(series, key=lambda l: (l.split()[0] != "Flat", l.split()[0],
                                          int(l.split("=")[1]) if "=" in l else 0))
    fig, (ax, bx, cx) = plt.subplots(1, 3, figsize=(15, 4.6))
    for lab in order:
        mk, ls, col = style(lab)
        ds_ = [d for d in dims if d in series[lab]]
        if not lab.startswith("Flat"):
            ax.plot(ds_, [_f(series[lab][d], "recall_at_10") for d in ds_], marker=mk, ls=ls, color=col, label=lab, ms=4)
        bx.plot(ds_, [_f(series[lab][d], "qps_median") for d in ds_], marker=mk, ls=ls, color=col, label=lab, ms=5 if mk != "*" else 9)
    for a in (ax, bx, cx):
        a.set_xscale("log", base=2)
        a.set_xticks(dims)
        a.set_xticklabels([str(d) for d in dims])
        a.axvline(CONTROL, color="0.6", lw=0.8, ls=":")
        a.grid(True, which="both", alpha=0.3)
        a.set_xlabel("dimensionality d (control = 128)")
    ax.set_ylabel(f"recall@10 ({int(rows[0]['n_queries']):,} queries)")
    ax.set_title("(a) recall at fixed search settings", fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    bx.set_yscale("log")
    bx.set_ylabel("QPS (1 thread, log scale)")
    bx.set_title("(b) single-query throughput (black star = exact search)", fontsize=10)
    sd = sorted(int(k) for k in stats)
    cx.plot(sd, [stats[str(d)]["median_relative_contrast"] - 1 for d in sd], "o-", color="#9467bd", label="relative contrast − 1 (median)")
    cx.fill_between(sd, [stats[str(d)]["rc_p10"] - 1 for d in sd], [stats[str(d)]["rc_p90"] - 1 for d in sd],
                    color="#9467bd", alpha=0.15, label="RC − 1, 10th–90th percentile")
    cx.plot(sd, [stats[str(d)]["cv_distance"] for d in sd], "s--", color="#8c564b", label="std / mean of all distances")
    cx.plot(sd, [stats[str(d)]["median_relative_gap_10_11"] for d in sd], "^-", color="#e377c2", label="(d11 − d10) / d10 (median)")
    cx.set_yscale("log")
    cx.set_title("(c) distance concentration", fontsize=10)
    cx.legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Part 4: uniform random unit vectors, n = {int(rows[0]['n_base']):,}; one property varied: d", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "figures" / "part4_dimensionality.png", dpi=160)
    plt.close(fig)
    print(f"figure: {out / 'figures' / 'part4_dimensionality.png'}")

    rec_tab, qps_tab = [], []
    for lab in order:
        s = series[lab]
        c = _f(s[CONTROL], "recall_at_10") if CONTROL in s else None
        rr = {"index_setting": lab}
        qr = {"index_setting": lab}
        for d in dims:
            if d in s:
                r = _f(s[d], "recall_at_10")
                rr[f"d={d}"] = f"{r:.4f}" + (f" ({r - c:+.4f})" if c is not None and d != CONTROL else "")
                qr[f"d={d}"] = f"{_f(s[d], 'qps_median'):.0f}"
        rec_tab.append(rr)
        qps_tab.append(qr)
    cols = ["index_setting"] + [f"d={d}" for d in dims]
    write_table(rec_tab, cols, out / "tables" / "part4_recall_by_d",
                "Part 4: recall@10 per d (in brackets: change vs the d = 128 control)")
    write_table(qps_tab, cols, out / "tables" / "part4_qps_by_d", "Part 4: single-thread QPS per d")
    keys = ["mean_distance", "std_distance", "cv_distance", "median_nn_distance", "median_d10",
            "median_relative_contrast", "rc_p10", "rc_p90", "median_d10_over_mean", "median_relative_gap_10_11"]
    st_tab = [{"d": d, **{k: f"{stats[str(d)][k]:.5f}" for k in keys},
               "boundary_ties": stats[str(d)]["boundary_ties"]["queries_with_boundary_tie"],
               "fingerprint": stats[str(d)]["fingerprint_sha256"][:16]} for d in sd]
    write_table(st_tab, ["d"] + keys + ["boundary_ties", "fingerprint"], out / "tables" / "part4_distance_stats",
                "Part 4: distance statistics per d (Euclidean distances; medians over queries)")
    print(f"tables: {out / 'tables'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
