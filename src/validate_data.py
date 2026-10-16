"""Inspect and validate the ann-benchmarks dataset files.

Writes one JSON report per dataset to results/metadata/dataset_<name>.json.

Usage:
  python -m src.validate_data                 # all datasets in the registry
  python -m src.validate_data --dataset sift
  python -m src.validate_data --allow-missing # exit 0 even if files are absent

Exit codes: 0 = all present datasets passed; 1 = a check failed; 2 = a dataset file is missing.
Memory: vectors are streamed in blocks; only queries, ground truth and a handful of
spot-check rows are held in RAM at once.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from src.config import DATASETS, METADATA_DIR, N_TUNE_QUERIES, SEED, DatasetSpec
from src.data_loader import chunked_stats, inspect_hdf5, l2_normalize, split_queries

N_SPOT_CHECK = 10       # queries whose stored ground-truth distances we recompute
NORM_TOL = 1e-4


def _check(checks: list, name: str, ok: bool, detail) -> None:
    checks.append({"check": name, "passed": bool(ok), "detail": detail})


def validate(spec: DatasetSpec, path: Path | None = None, n_spot: int = N_SPOT_CHECK) -> dict:
    path = Path(path) if path else spec.path
    report: dict = {"dataset": spec.name, "expected_path": str(path), "download_url": spec.url}
    if not path.exists():
        report["status"] = "MISSING"
        return report

    meta = inspect_hdf5(path)
    report["metadata"] = meta
    checks: list = []
    report["checks"] = checks
    ds = meta["datasets"]

    _check(checks, "required_keys_present", not meta["missing_keys"], meta["missing_keys"])
    if meta["missing_keys"]:
        report["status"] = "FAILED"
        return report

    n, d = ds["train"]["shape"]
    q, dq = ds["test"]["shape"]
    gq, gk = ds["neighbors"]["shape"]
    _check(checks, "train_shape", (n, d) == (spec.n_train, spec.dim), {"found": [n, d], "expected": [spec.n_train, spec.dim]})
    _check(checks, "train_dtype_float32", ds["train"]["dtype"] == "float32", ds["train"]["dtype"])
    _check(checks, "query_shape", (q, dq) == (spec.n_queries, spec.dim), {"found": [q, dq], "expected": [spec.n_queries, spec.dim]})
    _check(checks, "query_dtype_float32", ds["test"]["dtype"] == "float32", ds["test"]["dtype"])
    _check(checks, "gt_rows_match_queries", gq == q, {"neighbors_rows": gq, "queries": q})
    _check(checks, "gt_has_at_least_10_columns", gk >= 10, gk)
    _check(checks, "gt_distances_shape_matches", ds["distances"]["shape"] == [gq, gk], ds["distances"]["shape"])
    _check(checks, "neighbors_integer_dtype", np.issubdtype(np.dtype(ds["neighbors"]["dtype"]), np.integer), ds["neighbors"]["dtype"])
    _check(checks, "distance_attr_matches_spec", meta["attrs"].get("distance") == spec.source_metric,
           {"attr": meta["attrs"].get("distance"), "expected": spec.source_metric})

    with h5py.File(path, "r") as f:
        train_stats = chunked_stats(f["train"])
        query_stats = chunked_stats(f["test"])
        report["train_stats"], report["query_stats"] = train_stats, query_stats
        for label, st in (("train", train_stats), ("query", query_stats)):
            _check(checks, f"{label}_no_nan", st["nan_count"] == 0, st["nan_count"])
            _check(checks, f"{label}_no_inf", st["inf_count"] == 0, st["inf_count"])
            # Zero-norm rows are an anomaly worth recording; they only break things if we normalise.
            _check(checks, f"{label}_zero_norm_rows_recorded", True, st["zero_norm_rows"])

        gt_n = f["neighbors"][:]
        gt_d = f["distances"][:]
        _check(checks, "gt_ids_in_range", int(gt_n.min()) >= 0 and int(gt_n.max()) < n,
               {"min": int(gt_n.min()), "max": int(gt_n.max()), "n_train": n})
        # Ground truth should be ordered nearest-first. Allow tiny float noise.
        nondecreasing = np.all(np.diff(gt_d, axis=1) >= -1e-6, axis=1)
        _check(checks, "gt_distances_sorted_per_row", nondecreasing.all(), {"rows_not_sorted": int((~nondecreasing).sum())})
        # Exact ties at rank 10 matter for recall@10 later (Part 1); record how common they are.
        report["gt_ties_at_rank10"] = int((gt_d[:, 9] == gt_d[:, 10]).sum()) if gk > 10 else None

        # Spot-check: recompute distances to the stored neighbours for a few random queries.
        rng = np.random.default_rng(SEED)
        qids = np.sort(rng.choice(q, size=min(n_spot, q), replace=False))
        queries = f["test"][qids]
        max_err, rel = 0.0, []
        for row, qi in enumerate(qids):
            ids = gt_n[qi]
            order = np.argsort(ids)               # h5py fancy indexing needs increasing indices
            vecs = np.empty((gk, d), dtype=np.float32)
            vecs[order] = f["train"][ids[order]]
            qv = queries[row:row + 1]
            if spec.normalize:
                qv, _ = l2_normalize(qv)
                vecs, _ = l2_normalize(vecs)
                sq = ((vecs - qv) ** 2).sum(1)
                # For unit vectors ||q-x||^2 = 2 - 2cos = 2 * cosine_distance.
                recomputed = sq / 2
            else:
                recomputed = np.sqrt(((vecs - qv) ** 2).sum(1))
            err = np.abs(recomputed - gt_d[qi])
            max_err = max(max_err, float(err.max()))
            rel.append(bool(np.all(np.diff(recomputed) >= -1e-5)))
        report["spot_check"] = {"query_ids": qids.tolist(), "max_abs_error_vs_stored": max_err,
                                "relation": "sq_l2_normalised/2 == stored" if spec.normalize else "l2 == stored"}
        _check(checks, "spot_check_stored_distances_reproduced", max_err < 1e-3 * max(1.0, float(gt_d.max())), max_err)
        _check(checks, "spot_check_order_preserved_in_our_metric", all(rel), rel)

        if spec.normalize:
            sample = np.sort(rng.choice(n, size=min(10_000, n), replace=False))
            normed, zeros = l2_normalize(f["train"][sample])
            norms = np.linalg.norm(normed, axis=1)
            nz = norms[norms > 0]
            _check(checks, "normalised_sample_unit_norm", np.all(np.abs(nz - 1) < NORM_TOL),
                   {"sample_size": len(sample), "max_dev": float(np.abs(nz - 1).max()), "zero_rows_in_sample": zeros})

    split = split_queries(q, N_TUNE_QUERIES)
    sizes = (split.tune.stop - split.tune.start, split.eval.stop - split.eval.start)
    _check(checks, "query_split_5000_5000", sizes == (5000, 5000) and split.tune.stop == split.eval.start,
           {"tune": [split.tune.start, split.tune.stop], "eval": [split.eval.start, split.eval.stop]})

    report["status"] = "PASSED" if all(c["passed"] for c in checks) else "FAILED"
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASETS), action="append")
    ap.add_argument("--allow-missing", action="store_true")
    args = ap.parse_args(argv)

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    code = 0
    for name in args.dataset or list(DATASETS):
        rep = validate(DATASETS[name])
        (METADATA_DIR / f"dataset_{name}.json").write_text(json.dumps(rep, indent=2))
        print(f"[{rep['status']}] {name}: {rep['expected_path']}")
        if rep["status"] == "MISSING":
            print(f"    download from {rep['download_url']} and place it at the path above")
            code = max(code, 0 if args.allow_missing else 2)
        else:
            for c in rep["checks"]:
                if not c["passed"]:
                    print(f"    FAILED {c['check']}: {c['detail']}")
            code = max(code, 0 if rep["status"] == "PASSED" else 1)
    return code


if __name__ == "__main__":
    sys.exit(main())
