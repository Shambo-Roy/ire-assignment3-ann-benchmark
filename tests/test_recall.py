import numpy as np
import pytest

from src.recall import (REL_TOL, boundary_tie_stats, recall_at_k, reference_from_candidates,
                        true_sq_distances)

K = 10
TAU = np.array([10.0])                       # true 10th-nearest distance for the single test query
REF_IDS = np.arange(10)[None, :]             # reference top-10 = ids 0..9
REF_D = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=np.float64)


def r(ids, d, tau=TAU):
    return recall_at_k(np.array(ids, ndmin=2), np.array(d, dtype=np.float64, ndmin=2), tau, K)[0]


def test_perfect_result_is_one():
    assert r(REF_IDS, REF_D) == 1.0


def test_disjoint_far_result_is_zero():
    assert r(np.arange(100, 110), np.full(10, 50.0)) == 0.0


def test_order_does_not_matter():
    perm = np.random.default_rng(0).permutation(10)
    assert r(REF_IDS[0][perm], REF_D[0][perm]) == 1.0


def test_tied_non_reference_id_at_boundary_counts():
    # id 999 is not in the reference list but is exactly as far as the 10th neighbour
    ids = list(range(9)) + [999]
    d = list(range(1, 10)) + [10.0]
    assert r(ids, d) == 1.0


def test_tie_within_relative_tolerance_counts_but_beyond_does_not():
    ids = list(range(9)) + [999]
    inside = list(range(1, 10)) + [10.0 * (1 + 0.5 * REL_TOL)]
    outside = list(range(1, 10)) + [10.0 * (1 + 10 * REL_TOL)]
    assert r(ids, inside) == 1.0
    assert r(ids, outside) == pytest.approx(0.9)


def test_hand_computed_partial_case():
    # 7 true neighbours + 3 vectors farther than tau
    ids = list(range(7)) + [500, 501, 502]
    d = list(range(1, 8)) + [11.0, 12.0, 13.0]
    assert r(ids, d) == pytest.approx(0.7)


def test_duplicates_counted_once():
    ids = [0] * 10
    assert r(ids, [1.0] * 10) == pytest.approx(0.1)
    ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 8]
    assert r(ids, [1, 2, 3, 4, 5, 6, 7, 8, 9, 9]) == pytest.approx(0.9)


def test_padding_and_short_lists_keep_denominator_k():
    ids = [0, 1, 2, -1, -1, -1, -1, -1, -1, -1]
    d = [1, 2, 3] + [np.inf] * 7
    assert r(ids, d) == pytest.approx(0.3)
    # fewer than k columns returned at all
    assert recall_at_k(np.array([[0, 1]]), np.array([[1.0, 2.0]]), TAU, K)[0] == pytest.approx(0.2)


def test_minus_one_never_counts_even_with_small_distance():
    assert r([-1] + list(range(1, 10)), [0.0] + list(range(2, 11))) == pytest.approx(0.9)


def test_tau_zero_duplicate_query():
    # the query equals 10+ base vectors: tau = 0; exact zeros count, anything positive does not
    assert r(list(range(10)), [0.0] * 10, tau=np.array([0.0])) == 1.0
    assert r(list(range(10)), [0.0] * 9 + [1e-9], tau=np.array([0.0])) == pytest.approx(0.9)


def test_matches_id_intersection_when_no_ties():
    rng = np.random.default_rng(1)
    base = rng.normal(size=(2000, 16))
    q = rng.normal(size=(50, 16))
    d_all = ((q[:, None, :] - base[None]) ** 2).sum(-1)
    order = np.argsort(d_all, axis=1)
    true10 = order[:, :10]
    tau = np.take_along_axis(d_all, true10, 1)[:, -1]
    # a noisy "approximate" answer: swap some true neighbours for ranks 10..30
    approx = np.where(rng.random((50, 10)) < 0.3, order[:, 10:20], true10)
    dists = np.take_along_axis(d_all, approx, 1)
    ours = recall_at_k(approx, dists, tau, K)
    naive = np.array([len(set(a) & set(t)) / 10 for a, t in zip(approx, true10)])
    np.testing.assert_allclose(ours, naive)


def test_vectorised_rows_independent():
    ids = np.vstack([REF_IDS[0], np.arange(100, 110)])
    d = np.vstack([REF_D[0], np.full(10, 50.0)])
    np.testing.assert_allclose(recall_at_k(ids, d, np.array([10.0, 10.0]), K), [1.0, 0.0])


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        recall_at_k(np.zeros((1, 10), int), np.zeros((1, 9)), TAU, K)


# ------------------------------------------------------------ distance helpers

def test_true_sq_distances_float64_and_padding():
    base = np.array([[0, 0], [3, 4], [1, 1]], dtype=np.float32)
    q = np.array([[0, 0]], dtype=np.float32)
    d = true_sq_distances(q, base, np.array([[1, -1, 2]]))
    assert d.dtype == np.float64
    np.testing.assert_allclose(d, [[25.0, np.inf, 2.0]])


def test_reference_sorted_with_deterministic_tie_order():
    base = np.array([[1, 0], [0, 1], [2, 0], [-1, 0]], dtype=np.float32)   # ids 0,1,3 tie at distance 1
    q = np.zeros((1, 2), dtype=np.float32)
    ids, d = reference_from_candidates(q, base, np.array([[3, 2, 1, 0]]))
    np.testing.assert_array_equal(ids, [[0, 1, 3, 2]])
    np.testing.assert_allclose(d, [[1, 1, 1, 4]])


def test_boundary_tie_stats_counts_ties_and_gap():
    d = np.array([
        list(range(1, 11)) + [10, 10, 12],      # two extra vectors tied with the 10th
        list(range(1, 11)) + [11, 12, 13],      # no tie, gap 10%
    ], dtype=np.float64)
    st = boundary_tie_stats(d, K)
    assert st["queries_with_boundary_tie"] == 1
    assert st["max_extra_tied_vectors"] == 2
    assert st["queries_with_exact_equal_k_kplus1"] == 1
    assert st["min_positive_relative_gap"] == pytest.approx(0.1)
    assert st["queries_tied_by_tolerance"] == {"0": 1, "1e-07": 1, "1e-06": 1, "1e-05": 1, "0.0001": 1}


def test_boundary_tie_stats_near_tie_depends_on_tolerance():
    d = np.array([list(range(1, 11)) + [10 * (1 + 5e-7), 12, 13]], dtype=np.float64)
    st = boundary_tie_stats(d, K)
    assert st["queries_with_boundary_tie"] == 1                 # within rel_tol = 1e-6
    assert st["queries_with_exact_equal_k_kplus1"] == 0
    assert st["queries_tied_by_tolerance"]["1e-07"] == 0
    assert st["queries_tied_by_tolerance"]["1e-06"] == 1
