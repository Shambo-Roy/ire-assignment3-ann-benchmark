import faiss
import numpy as np
import pytest

from src.indexes import BuildSpec, CoarseQuantizerCache, build, index_size_bytes, set_search_param
from src.pareto import pareto_front, select_for_target
from src.results_io import read_rows, replace_rows


# ------------------------------------------------------------ pareto / selection

def P(r, q, tag=""):
    return {"recall": r, "qps": q, "tag": tag}


def test_pareto_front_removes_dominated_points():
    pts = [P(0.5, 1000, "a"), P(0.8, 500, "b"), P(0.7, 400, "dominated"), P(0.95, 100, "c"), P(0.6, 900, "d")]
    front = pareto_front(pts)
    assert [p["tag"] for p in front] == ["a", "d", "b", "c"]


def test_pareto_front_equal_recall_keeps_faster():
    front = pareto_front([P(0.9, 100, "slow"), P(0.9, 200, "fast")])
    assert [p["tag"] for p in front] == ["fast"]


def test_select_for_target_fastest_meeting_target_or_none():
    pts = [P(0.89, 2000), P(0.91, 800, "x"), P(0.96, 300, "y"), P(0.99, 100)]
    assert select_for_target(pts, 0.90)["tag"] == "x"
    assert select_for_target(pts, 0.95)["tag"] == "y"
    assert select_for_target([P(0.8, 10)], 0.95) is None


def test_target_selection_is_always_on_the_front():
    rng = np.random.default_rng(0)
    pts = [P(float(r), float(q), i) for i, (r, q) in enumerate(zip(rng.random(200), rng.random(200) * 1e4))]
    front_tags = {p["tag"] for p in pareto_front(pts)}
    for t in (0.5, 0.9, 0.95):
        assert select_for_target(pts, t)["tag"] in front_tags


# ------------------------------------------------------------ index builders

@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(1)
    centers = rng.normal(size=(20, 16)) * 5
    x = (centers[rng.integers(0, 20, 4000)] + rng.normal(size=(4000, 16))).astype(np.float32)
    q = (centers[rng.integers(0, 20, 50)] + rng.normal(size=(50, 16))).astype(np.float32)
    flat = faiss.IndexFlatL2(16)
    flat.add(x)
    return x, q, flat.search(q, 10)[1]


def _overlap(a, b):
    return np.mean([len(set(r) & set(s)) / 10 for r, s in zip(a, b)])


@pytest.mark.parametrize("spec,high", [
    (BuildSpec.make("HNSW", M=8, efC=40), 256),
    (BuildSpec.make("IVF-Flat", nlist=16), 16),
    (BuildSpec.make("IVF-PQ", nlist=16, m=8, nbits=8), 16),
])
def test_builders_produce_working_indexes_and_param_controls_recall(data, spec, high):
    x, q, gt = data
    res = build(spec, x)
    assert res.index.ntotal == len(x) and res.build_time_s > 0
    set_search_param(res.index, spec, 1)
    low_rec = _overlap(res.index.search(q, 10)[1], gt)
    set_search_param(res.index, spec, high)
    high_rec = _overlap(res.index.search(q, 10)[1], gt)
    assert high_rec >= low_rec
    if spec.index_type != "IVF-PQ":            # PQ is lossy; exhaustive IVF-Flat / big ef ~ exact
        assert high_rec > 0.95
    assert index_size_bytes(res.index) > 0


def test_shared_coarse_quantizer_matches_standalone_training(data):
    x, q, _ = data
    cache = CoarseQuantizerCache(x)
    shared = build(BuildSpec.make("IVF-Flat", nlist=16), x, cache)
    again = build(BuildSpec.make("IVF-Flat", nlist=16), x, CoarseQuantizerCache(x))
    for idx in (shared.index, again.index):
        idx.nprobe = 2
    np.testing.assert_array_equal(shared.index.search(q, 10)[1], again.index.search(q, 10)[1])
    pq = build(BuildSpec.make("IVF-PQ", nlist=16, m=8, nbits=8), x, cache)
    # the k-means time is charged to every IVF build that uses the centroids
    assert pq.timings["coarse_kmeans_s"] == shared.timings["coarse_kmeans_s"]


def test_pq_m_must_divide_dimension(data):
    with pytest.raises(ValueError):
        build(BuildSpec.make("IVF-PQ", nlist=16, m=5, nbits=8), data[0])


def test_ivfpq_is_smaller_than_ivfflat(data):
    x = data[0]
    flat = index_size_bytes(build(BuildSpec.make("IVF-Flat", nlist=16), x).index)
    pq = index_size_bytes(build(BuildSpec.make("IVF-PQ", nlist=16, m=4, nbits=8), x).index)
    assert pq < flat


def test_buildspec_name_and_roundtrip():
    s = BuildSpec.make("IVF-PQ", nlist=1024, m=16, nbits=8)
    assert s.name == "IVF-PQ_m16_nbits8_nlist1024" and s.search_param == "nprobe"
    assert BuildSpec.make("IVF-PQ", **s.p) == s


# ------------------------------------------------------------ results key

def test_replace_rows_custom_key_keeps_other_index_types(tmp_path):
    p = tmp_path / "r.csv"
    k = ("experiment_id", "dataset_id", "n_base", "index_type")
    replace_rows([{"experiment_id": "p2", "dataset_id": "s", "n_base": 1, "index_type": "HNSW", "qps_median": 1}], p, k)
    replace_rows([{"experiment_id": "p2", "dataset_id": "s", "n_base": 1, "index_type": "IVF-Flat", "qps_median": 2}], p, k)
    replace_rows([{"experiment_id": "p2", "dataset_id": "s", "n_base": 1, "index_type": "HNSW", "qps_median": 3}], p, k)
    got = {r["index_type"]: r["qps_median"] for r in read_rows(p)}
    assert got == {"HNSW": "3", "IVF-Flat": "2"}
