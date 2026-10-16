"""Fixtures that write tiny HDF5 files in the ann-benchmarks layout.

The real datasets are ~0.5 GB each, so tests use synthetic stand-ins with the same keys,
dtypes and attributes. 10,000 queries are kept so the 5,000/5,000 split is exercised exactly.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from src.config import DatasetSpec

N_TRAIN, DIM, N_Q, K_GT = 300, 8, 10_000, 100


def _ground_truth(train: np.ndarray, queries: np.ndarray, metric: str):
    if metric == "euclidean":
        d = np.sqrt(((queries[:, None, :] - train[None, :, :]) ** 2).sum(-1))
    else:  # angular: cosine distance on the raw (unnormalised) vectors, as ann-benchmarks does
        qn = queries / np.linalg.norm(queries, axis=1, keepdims=True)
        tn = train / np.linalg.norm(train, axis=1, keepdims=True)
        d = 1 - qn @ tn.T
    idx = np.argsort(d, axis=1, kind="stable")[:, :K_GT]
    return idx.astype(np.int32), np.take_along_axis(d, idx, axis=1).astype(np.float32)


def write_fake(path, metric: str, seed: int = 0, train=None, queries=None):
    rng = np.random.default_rng(seed)
    train = rng.normal(size=(N_TRAIN, DIM)).astype(np.float32) if train is None else train
    queries = rng.normal(size=(N_Q, DIM)).astype(np.float32) if queries is None else queries
    nb, dist = _ground_truth(train.astype(np.float64), queries.astype(np.float64), metric)
    with h5py.File(path, "w") as f:
        f.attrs["distance"] = metric
        f.attrs["type"] = "dense"
        f.create_dataset("train", data=train)
        f.create_dataset("test", data=queries)
        f.create_dataset("neighbors", data=nb)
        f.create_dataset("distances", data=dist)
    return path


def fake_spec(name: str, path, metric: str, normalize: bool) -> DatasetSpec:
    return DatasetSpec(name=name, filename=str(path), url="(test)", dim=DIM, n_train=N_TRAIN,
                       n_queries=N_Q, source_metric=metric, normalize=normalize)


@pytest.fixture(scope="session")
def euclid_file(tmp_path_factory):
    return write_fake(tmp_path_factory.mktemp("d") / "fake-euclidean.hdf5", "euclidean")


@pytest.fixture(scope="session")
def angular_file(tmp_path_factory):
    return write_fake(tmp_path_factory.mktemp("d") / "fake-angular.hdf5", "angular", seed=1)
