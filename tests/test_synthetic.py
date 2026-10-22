import numpy as np
import pytest

from src.recall import reference_from_candidates
from src.synthetic import distance_stats, generate


def test_generator_is_deterministic_and_seed_dependent():
    a, b = generate(32, 500, 50, seed=7), generate(32, 500, 50, seed=7)
    np.testing.assert_array_equal(a.base, b.base)
    np.testing.assert_array_equal(a.queries, b.queries)
    assert a.fingerprint() == b.fingerprint()
    assert generate(32, 500, 50, seed=8).fingerprint() != a.fingerprint()


def test_each_dimension_has_its_own_stream_and_queries_are_independent_of_base():
    a, b = generate(32, 500, 50), generate(64, 500, 50)
    assert a.fingerprint() != b.fingerprint()
    # queries are not simply the first base rows (separate child streams)
    assert not np.allclose(a.queries[:5], a.base[:5])
    # generating a different n_base does not change the queries of the same d
    np.testing.assert_array_equal(generate(32, 800, 50).queries, a.queries)


@pytest.mark.parametrize("d", [32, 128, 512])
def test_unit_norm_float32_and_shape(d):
    s = generate(d, 300, 20)
    assert s.base.shape == (300, d) and s.queries.shape == (20, d)
    assert s.base.dtype == np.float32 and s.base.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(np.linalg.norm(s.base, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(s.queries, axis=1), 1.0, atol=1e-5)
    assert s.dataset_id == f"synth_sphere_d{d}_n300_s12345"


def test_distance_stats_match_brute_force_and_concentrate_with_d():
    out = {}
    for d in (8, 256):
        s = generate(d, 2000, 30)
        full = np.sqrt(((s.queries[:, None, :].astype(np.float64) - s.base[None].astype(np.float64)) ** 2).sum(-1))
        order = np.argsort(full ** 2, axis=1)[:, :20]
        _, ref_d = reference_from_candidates(s.queries, s.base, order)
        st = distance_stats(s.queries, s.base, ref_d, 10, chunk=7)
        assert st["mean_distance"] == pytest.approx(full.mean(), rel=1e-9)
        assert st["std_distance"] == pytest.approx(full.std(), rel=1e-6)
        assert st["median_nn_distance"] == pytest.approx(np.median(np.sort(full, axis=1)[:, 0]), rel=1e-6)
        out[d] = st
    # random unit vectors: mean distance -> sqrt(2), relative spread and contrast shrink with d
    assert abs(out[256]["mean_distance"] - np.sqrt(2)) < 0.01
    assert out[256]["cv_distance"] < out[8]["cv_distance"]
    assert out[256]["median_relative_contrast"] < out[8]["median_relative_contrast"]
