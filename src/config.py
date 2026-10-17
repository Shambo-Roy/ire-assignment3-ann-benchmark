"""Central configuration: paths, dataset registry and fixed experimental constants.

Paths never depend on the author's machine. The repository is the ``code/`` folder of the
submission, and data and results live next to it, matching the zip layout:

    <submission>/code/      this git repository
    <submission>/data/      datasets (ANN_DATA_DIR overrides; never zipped)
    <submission>/results/   results.csv, figures/, metadata/ (ANN_RESULTS_DIR overrides)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _dir_from_env(var: str, default: Path) -> Path:
    env = os.environ.get(var)
    return Path(env).expanduser().resolve() if env else default.resolve()


RESULTS_DIR = _dir_from_env("ANN_RESULTS_DIR", REPO_ROOT.parent / "results")
FIGURES_DIR = RESULTS_DIR / "figures"
METADATA_DIR = RESULTS_DIR / "metadata"

# Global seed for anything stochastic (k-means training samples, synthetic data, ...).
SEED = 12345

# Query split mandated by the assignment: first half for tuning, second half for final numbers.
N_QUERIES = 10_000
N_TUNE_QUERIES = 5_000

# Recall is always measured at k = 10.
K = 10


def data_dir() -> Path:
    return _dir_from_env("ANN_DATA_DIR", REPO_ROOT.parent / "data")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    filename: str
    url: str
    dim: int
    n_train: int
    n_queries: int
    source_metric: str   # metric used by ann-benchmarks to compute the stored ground truth
    normalize: bool      # whether we L2-normalise vectors before using Euclidean distance

    @property
    def path(self) -> Path:
        return data_dir() / self.filename


# Expected values follow the assignment text / ann-benchmarks documentation.
# They are verified against the actual files by src/validate_data.py, not assumed.
DATASETS: dict[str, DatasetSpec] = {
    "sift": DatasetSpec(
        name="sift",
        filename="sift-128-euclidean.hdf5",
        url="http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        dim=128,
        n_train=1_000_000,
        n_queries=N_QUERIES,
        source_metric="euclidean",
        normalize=False,
    ),
    "glove": DatasetSpec(
        name="glove",
        filename="glove-100-angular.hdf5",
        url="http://ann-benchmarks.com/glove-100-angular.hdf5",
        dim=100,
        n_train=1_183_514,
        n_queries=N_QUERIES,
        source_metric="angular",
        normalize=True,
    ),
}
