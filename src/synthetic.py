"""Seeded synthetic data for Part 4: random vectors on the unit sphere.

Generation principle (identical for every dimensionality d):
  x ~ N(0, I_d) in float32, then L2-normalised, i.e. uniform on the unit sphere S^(d-1)
  ("random normalized vectors" in hypothesis 4.1). Base and query vectors are drawn
  independently from the same distribution.

Seed policy: one master seed (config.SEED). For each d, SeedSequence([SEED, d]) is split into two
child streams, one for the base vectors and one for the queries. The same (SEED, d, n) always gives
byte-identical data, a setting does not depend on which other settings are generated, and the
queries are independent of the base set.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from src.config import SEED
from src.data_loader import l2_normalize


@dataclass
class SyntheticSet:
    d: int
    seed: int
    base: np.ndarray       # (n, d) float32, unit norm
    queries: np.ndarray    # (q, d) float32, unit norm

    @property
    def dataset_id(self) -> str:
        return f"synth_sphere_d{self.d}_n{len(self.base)}_s{self.seed}"

    def fingerprint(self) -> str:
        """SHA-256 of the exact bytes, recorded so that reruns can be checked as identical."""
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.base).tobytes())
        h.update(np.ascontiguousarray(self.queries).tobytes())
        return h.hexdigest()


def generate(d: int, n_base: int = 100_000, n_query: int = 1_000, seed: int = SEED) -> SyntheticSet:
    base_seq, query_seq = np.random.SeedSequence([seed, d]).spawn(2)
    base = np.random.default_rng(base_seq).standard_normal((n_base, d), dtype=np.float32)
    queries = np.random.default_rng(query_seq).standard_normal((n_query, d), dtype=np.float32)
    base, zb = l2_normalize(base)
    queries, zq = l2_normalize(queries)
    if zb or zq:                       # probability ~0 for Gaussian draws; never hide it
        raise RuntimeError(f"zero-norm synthetic vectors (base {zb}, queries {zq})")
    return SyntheticSet(d, seed, np.ascontiguousarray(base), np.ascontiguousarray(queries))


def distance_stats(queries: np.ndarray, base: np.ndarray, ref_d: np.ndarray, k: int = 10,
                   chunk: int = 100) -> dict:
    """Distance-concentration statistics for hypothesis 4.1 (Euclidean distances, not squared).

    Over ALL query-base pairs (exactly, in chunks): mean and std of ||q - x||.
    Per query, from the exact float64 reference ref_d (squared distances, sorted):
      relative contrast RC = mean distance / nearest-neighbour distance,
      ratio d_k / mean distance, relative gap (d_{k+1} - d_k) / d_k.
    Medians over queries are reported, plus the RC 10th/90th percentiles.
    """
    nq = len(queries)
    s = s2 = 0.0
    per_q_mean = np.empty(nq)
    base_t = base.T.astype(np.float64)                       # converted once (410 MB at d = 512)
    xn = (base_t ** 2).sum(axis=0)                           # actual norms (float32 normalisation is not exact)
    for i in range(0, nq, chunk):
        qb = queries[i:i + chunk].astype(np.float64)
        g = qb @ base_t
        dist = np.sqrt(np.maximum((qb ** 2).sum(axis=1)[:, None] + xn[None, :] - 2.0 * g, 0.0))
        s += dist.sum()
        s2 += (dist ** 2).sum()
        per_q_mean[i:i + chunk] = dist.mean(axis=1)
    n = nq * len(base)
    mean = s / n
    std = float(np.sqrt(max(s2 / n - mean ** 2, 0.0)))
    d1 = np.sqrt(ref_d[:, 0])
    dk = np.sqrt(ref_d[:, k - 1])
    dk1 = np.sqrt(ref_d[:, k])
    rc = per_q_mean / np.maximum(d1, 1e-12)
    return {
        "mean_distance": float(mean), "std_distance": std, "cv_distance": std / float(mean),
        "median_nn_distance": float(np.median(d1)), "median_d10": float(np.median(dk)),
        "median_relative_contrast": float(np.median(rc)),
        "rc_p10": float(np.percentile(rc, 10)), "rc_p90": float(np.percentile(rc, 90)),
        "median_d10_over_mean": float(np.median(dk / per_q_mean)),
        "median_relative_gap_10_11": float(np.median((dk1 - dk) / np.maximum(dk, 1e-12))),
        "n_pairs": int(n),
    }
