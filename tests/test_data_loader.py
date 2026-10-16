import h5py
import numpy as np
import pytest

from src.config import DATASETS, data_dir
from src.data_loader import inspect_hdf5, l2_normalize, load_dataset, split_queries
from src.validate_data import validate
from tests.conftest import DIM, N_Q, N_TRAIN, fake_spec, write_fake


# ----------------------------------------------------------------- metadata

def test_inspect_reports_shapes_and_attrs(euclid_file):
    info = inspect_hdf5(euclid_file)
    assert info["missing_keys"] == []
    assert info["datasets"]["train"]["shape"] == [N_TRAIN, DIM]
    assert info["datasets"]["test"]["shape"] == [N_Q, DIM]
    assert info["datasets"]["neighbors"]["shape"] == [N_Q, 100]
    assert info["datasets"]["train"]["dtype"] == "float32"
    assert info["attrs"]["distance"] == "euclidean"
    assert info["file_size_bytes"] > 0


def test_inspect_flags_missing_keys(tmp_path):
    p = tmp_path / "bad.hdf5"
    with h5py.File(p, "w") as f:
        f.create_dataset("train", data=np.zeros((3, 2), np.float32))
    assert set(inspect_hdf5(p)["missing_keys"]) == {"test", "neighbors", "distances"}


def test_registry_paths_are_relative_to_repo_not_machine(monkeypatch, tmp_path):
    monkeypatch.setenv("ANN_DATA_DIR", str(tmp_path))
    assert data_dir() == tmp_path.resolve()
    assert DATASETS["sift"].path == tmp_path.resolve() / "sift-128-euclidean.hdf5"


# ----------------------------------------------------------------- validation

def test_validate_passes_on_wellformed_euclidean(euclid_file):
    rep = validate(fake_spec("e", euclid_file, "euclidean", False), euclid_file)
    failed = [c for c in rep["checks"] if not c["passed"]]
    assert rep["status"] == "PASSED", failed


def test_validate_passes_on_wellformed_angular_with_normalisation(angular_file):
    rep = validate(fake_spec("a", angular_file, "angular", True), angular_file)
    failed = [c for c in rep["checks"] if not c["passed"]]
    assert rep["status"] == "PASSED", failed
    assert rep["spot_check"]["max_abs_error_vs_stored"] < 1e-4


def test_validate_reports_missing_file(tmp_path):
    rep = validate(fake_spec("m", tmp_path / "nope.hdf5", "euclidean", False), tmp_path / "nope.hdf5")
    assert rep["status"] == "MISSING"


def test_validate_detects_nan_and_wrong_shape(tmp_path):
    train = np.random.default_rng(0).normal(size=(N_TRAIN, DIM)).astype(np.float32)
    train[5, 3] = np.nan
    p = write_fake(tmp_path / "nan.hdf5", "euclidean", train=train)
    rep = validate(fake_spec("n", p, "euclidean", False), p)
    by_name = {c["check"]: c for c in rep["checks"]}
    assert rep["status"] == "FAILED"
    assert not by_name["train_no_nan"]["passed"]

    spec_wrong_dim = fake_spec("w", p, "euclidean", False).__class__(
        name="w", filename=str(p), url="", dim=DIM + 1, n_train=N_TRAIN, n_queries=N_Q,
        source_metric="euclidean", normalize=False)
    by_name = {c["check"]: c for c in validate(spec_wrong_dim, p)["checks"]}
    assert not by_name["train_shape"]["passed"]


def test_validate_detects_wrong_metric_attribute(euclid_file):
    rep = validate(fake_spec("x", euclid_file, "angular", False), euclid_file)
    by_name = {c["check"]: c for c in rep["checks"]}
    assert not by_name["distance_attr_matches_spec"]["passed"]


# ----------------------------------------------------------------- split

def test_split_is_5000_5000_disjoint_and_ordered():
    s = split_queries(10_000)
    tune, ev = np.arange(10_000)[s.tune], np.arange(10_000)[s.eval]
    assert len(tune) == len(ev) == 5000
    assert tune[0] == 0 and tune[-1] == 4999 and ev[0] == 5000 and ev[-1] == 9999
    assert np.intersect1d(tune, ev).size == 0


@pytest.mark.parametrize("n_tune", [0, 10_000, -1, 20_000])
def test_split_rejects_degenerate(n_tune):
    with pytest.raises(ValueError):
        split_queries(10_000, n_tune)


def test_loaded_dataset_exposes_matching_query_and_gt_halves(euclid_file):
    ds = load_dataset(fake_spec("e", euclid_file, "euclidean", False), euclid_file)
    assert ds.tune_queries.shape == (5000, DIM) and ds.eval_queries.shape == (5000, DIM)
    assert ds.tune_gt.shape == (5000, 100) and ds.eval_gt.shape == (5000, 100)
    np.testing.assert_array_equal(ds.eval_queries[0], ds.queries[5000])
    np.testing.assert_array_equal(ds.eval_gt[0], ds.gt_neighbors[5000])


# ----------------------------------------------------------------- normalisation

def test_l2_normalize_unit_rows_and_zero_rows_counted():
    x = np.array([[3, 4], [0, 0], [1e-3, 0], [-2, 0]], dtype=np.float32)
    original = x.copy()
    y, zeros = l2_normalize(x)
    np.testing.assert_array_equal(x, original)          # input untouched
    assert zeros == 1
    np.testing.assert_allclose(np.linalg.norm(y[[0, 2, 3]], axis=1), 1.0, atol=1e-6)
    np.testing.assert_array_equal(y[1], [0, 0])         # zero row stays zero, not NaN
    np.testing.assert_allclose(y[0], [0.6, 0.8], atol=1e-6)
    assert y.dtype == np.float32


def test_normalised_euclidean_ranking_equals_cosine_ranking():
    rng = np.random.default_rng(3)
    x, q = rng.normal(size=(500, 20)), rng.normal(size=(1, 20))
    cos = (x @ q[0]) / (np.linalg.norm(x, axis=1) * np.linalg.norm(q))
    xn, _ = l2_normalize(x)
    qn, _ = l2_normalize(q)
    l2 = ((xn - qn) ** 2).sum(1)
    np.testing.assert_array_equal(np.argsort(-cos), np.argsort(l2))
    np.testing.assert_allclose(l2, 2 - 2 * cos, atol=1e-5)


def test_load_dataset_normalises_only_when_spec_says_so(angular_file):
    raw = load_dataset(fake_spec("a", angular_file, "angular", False), angular_file)
    nrm = load_dataset(fake_spec("a", angular_file, "angular", True), angular_file)
    assert not np.allclose(np.linalg.norm(raw.train, axis=1), 1.0)
    np.testing.assert_allclose(np.linalg.norm(nrm.train, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(nrm.queries, axis=1), 1.0, atol=1e-5)
    assert nrm.train.dtype == np.float32 and nrm.train.flags["C_CONTIGUOUS"]
    # the file on disk is unchanged
    with h5py.File(angular_file, "r") as f:
        np.testing.assert_array_equal(f["train"][:], raw.train)
