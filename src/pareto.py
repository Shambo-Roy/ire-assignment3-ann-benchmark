"""Selection rules for Part 2, applied to TUNING measurements only.

pareto_front: configurations not dominated in (recall, qps); higher is better for both.
select_for_target: the fastest configuration whose tuning recall reaches the target.
"""

from __future__ import annotations


def pareto_front(points: list[dict], recall_key: str = "recall", qps_key: str = "qps") -> list[dict]:
    """Return the non-dominated points, sorted by increasing recall.

    A point is dominated if another point has recall >= and qps >= with at least one strict.
    Exact duplicates in both metrics are kept once (the first seen).
    """
    pts = sorted(points, key=lambda p: (-p[recall_key], -p[qps_key]))
    front, best_qps = [], float("-inf")
    for p in pts:                      # walk from highest recall downwards
        if p[qps_key] > best_qps:
            front.append(p)
            best_qps = p[qps_key]
    return sorted(front, key=lambda p: p[recall_key])


def select_for_target(points: list[dict], target: float, recall_key: str = "recall",
                      qps_key: str = "qps") -> dict | None:
    ok = [p for p in points if p[recall_key] >= target]
    return max(ok, key=lambda p: p[qps_key]) if ok else None
