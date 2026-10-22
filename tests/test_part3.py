import faiss
import numpy as np
import pytest

from src.indexes import BuildSpec, add_exact_refine, build, index_size_bytes
from src.recall import pool_recall, recall_at_k, reference_from_candidates, true_sq_distances

TAU = np.array([10.0])


def pr(ids, d, tau=TAU):
    return pool_recall(np.array(ids, ndmin=2), np.array(d, dtype=np.float64, ndmin=2), tau, 10)[0]


# ------------------------------------------------------------ pool recall (recall 10@100)

def test_pool_recall_counts_true_neighbours_anywhere_in_pool():
    ids = list(range(100, 190)) + list(range(10))          # the 10 true neighbours sit at the END of a 100-pool
    d = [50.0] * 90 + list(range(1, 11))
    assert pr(ids, d) == 1.0
    # recall@10 of the same list (first 10 only) is 0: re-ranking is what would recover them
    assert recall_at_k(np.array([ids]), np.array([d], float), TAU, 10)[0] == 0.0


def test_pool_recall_partial_and_capped_at_one_with_ties():
    assert pr(list(range(7)) + list(range(100, 193)), list(range(1, 8)) + [50.0] * 93) == pytest.approx(0.7)
    # 12 candidates all tied at tau: capped at 10/10
    assert pr(list(range(12)) + list(range(100, 188)), [10.0] * 12 + [50.0] * 88) == 1.0


def test_pool_recall_ignores_duplicates_and_padding():
    ids = [0] * 50 + [-1] * 50
    d = [1.0] * 50 + [np.inf] * 50
    assert pr(ids, d) == pytest.approx(0.1)


# ------------------------------------------------------------ exact re-ranking

@pytest.fixture(scope="module")
def built():
    rng = np.random.default_rng(3)
    centers = rng.normal(size=(30, 16)) * 4
    x = (centers[rng.integers(0, 30, 6000)] + rng.normal(size=(6000, 16))).astype(np.float32)
    q = (centers[rng.integers(0, 30, 60)] + rng.normal(size=(60, 16))).astype(np.float32)
    res = build(BuildSpec.make("IVF-PQ", nlist=32, m=4, nbits=8), x)
    refine, _ = add_exact_refine(res.index, x, k_factor=10)
    flat = faiss.IndexFlatL2(16)
    flat.add(x)
    _, cand = flat.search(q, 30)
    _, ref_d = reference_from_candidates(q, x, cand)
    return x, q, res, refine, ref_d[:, 9]


def test_refine_wrapper_shares_base_and_counts_all_vectors(built):
    x, _, res, refine, _ = built
    assert refine.ntotal == res.index.ntotal == len(x)
    assert index_size_bytes(refine) > index_size_bytes(res.index) + x.nbytes * 0.99   # raw vectors stored


@pytest.mark.parametrize("nprobe", [1, 4, 32])
def test_reranked_recall_equals_pool_recall_and_never_below_plain(built, nprobe):
    x, q, res, refine, tau = built
    res.index.nprobe = nprobe
    plain = res.index.search(q, 10)[1]
    rer = refine.search(q, 10)[1]
    pool = res.index.search(q, 100)[1]
    r_plain = recall_at_k(plain, true_sq_distances(q, x, plain), tau).mean()
    r_rer = recall_at_k(rer, true_sq_distances(q, x, rer), tau).mean()
    r_pool = pool_recall(pool, true_sq_distances(q, x, pool), tau).mean()
    # exact re-ranking picks the true best 10 of the pool: it reaches the pool bound (up to float32 ties)
    assert r_rer == pytest.approx(r_pool, abs=0.02)
    assert r_rer >= r_plain - 1e-12
    assert r_pool >= r_plain - 1e-12


def test_refine_rejects_wrong_shaped_vectors_before_faiss_reads_them(built):
    x, _, res, _, _ = built
    for bad in (np.zeros((0, 16), np.float32), x[:-1], x[:, :8]):
        with pytest.raises(ValueError):
            add_exact_refine(res.index, bad)
