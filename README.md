# IRE Assignment 3: Benchmarking Approximate Nearest Neighbour Search

CS4.406 Information Retrieval & Extraction. This repo benchmarks exact search, HNSW, IVF-Flat and
IVF-PQ (FAISS) on SIFT-1M and GloVe-100, then runs a compression experiment (Part 3) and a
synthetic dimensionality experiment (Part 4).

## Current status

- **Milestone 1 (environment + data validation): COMPLETED.** Both datasets are downloaded and validated.
- **Milestone 2 (Part 1 exact baseline): COMPLETED** on 2026-10-17. Results: `../results/results.csv` (13 rows, commit `e098ca9`).
- **Milestone 3 (Part 2 ANN benchmarks): COMPLETED** on 2026-10-17. 367 rows, 2 figures, 3 tables; plus a build-scaling check.
- **Hypotheses for Parts 3 and 4:** written by the author and committed in `55016fe` (pushed) before any Part 3/4 run.
- **Milestone 4 (Part 3 compression): COMPLETED** on 2026-10-22. 158 rows (guarded by hypothesis `55016fe`), main figure, 2 tables.
- **Milestone 5 (Part 4 dimensionality, synthetic data): COMPLETED** on 2026-10-22. 55 rows (guarded by `55016fe`), main figure, 3 tables.
- **Milestone 6 (report and submission): COMPLETED** on 2026-10-30. `bash run_all.sh --quick` took 4 min 19 s on the dev desktop. See `IMPLEMENTATION_LOG.md`.

Part 1 headline numbers (eval queries, 1 thread pinned to a P-core, median of 5):

| | SIFT-1M | GloVe-100 |
|---|---|---|
| Exact, one query at a time | 38.05 QPS (26.3 ms) | 39.98 QPS (25.0 ms) |
| Exact, batch of 5,000 (BLAS path) | 325.97 QPS (8.57×) | 334.04 QPS (8.36×) |
| Stored ground truth vs our exact search, recall@10 | 1.000 | 1.000 |
| Queries with a tie at rank 10 (rel_tol 1e-6) | 137 | 2 |

Part 2 headline numbers (configuration chosen on tune queries; eval recall / single-thread QPS):

| Index | SIFT @0.95 | GloVe @0.95 | Max recall reached (SIFT / GloVe) |
|---|---|---|---|
| HNSW (M=32) | 0.971 / 6,783 QPS | 0.955 / 762.5 QPS | 0.996 / 0.984 |
| IVF-Flat (nlist=16384) | 0.973 / 1,709 QPS | 0.956 / 412.1 QPS | 0.999 / 0.997 |
| IVF-PQ | not reached | not reached | 0.884 (m=64) / 0.869 (m=50) |

Figures: `../results/figures/part2_{sift,glove}_recall_vs_qps.png`. Tables: `../results/tables/part2_*.md`.

Part 3 headline numbers (SIFT-1M, IVF-PQ nlist 4096; eval):

| m (code bytes) | B/vector plain / + re-rank | Best recall plain | Best recall + re-rank top 100 | QPS @0.95, re-ranked |
|---|---|---|---|---|
| 8 | 18.3 / 530.3 | 0.415 | 0.888 | not reached |
| 16 | 26.3 / 538.3 | 0.597 | 0.986 | 2,663 |
| 32 | 42.3 / 554.3 | 0.746 | 0.999 | 1,637 |
| 64 | 74.3 / 586.3 | 0.886 | 0.999 | 931 |

Re-ranked recall equals the recall 10@100 of the candidate pool in every row. Re-ranking costs about 0.02 ms per query.
Figure: `../results/figures/part3_compression.png`.

Part 4 headline numbers (seeded random unit vectors, n = 100,000, 1,000 queries; fixed settings):

| d | Relative contrast | HNSW ef=256 recall | IVF-Flat nprobe=256 recall | Better at equal QPS |
|---|---|---|---|---|
| 32 | 1.755 | 0.996 | 0.978 | HNSW |
| 64 | 1.425 | 0.873 | 0.870 | IVF-Flat |
| 128 (control) | 1.261 | 0.576 | 0.736 | IVF-Flat |
| 256 | 1.167 | 0.351 | 0.630 | IVF-Flat |
| 512 | 1.111 | 0.231 | 0.529 | IVF-Flat |

Figure: `../results/figures/part4_dimensionality.png`. Generator: `src/synthetic.py` (seeded, with data fingerprints).

## Directory layout

The submission root holds four items: `report.pdf`, `code/`, `results/` and `ai_log/`. Code writes its outputs
straight to the top-level `results/`, so there are no copies inside the repo.

```
Assignment 3/                    # submission root
├── data/                        # datasets: NOT submitted, never committed (ANN_DATA_DIR overrides)
├── report.pdf                   # (planned)
├── results/                     # written by code/ (ANN_RESULTS_DIR overrides)
│   ├── metadata/                # environment.json, dataset_<name>.json
│   ├── results.csv              # (planned) one row per benchmark run
│   └── figures/                 # (planned)
├── ai_log/                      # ai_statement.md, prompts/responses, transcripts/
└── code/                        # THIS git repository
    ├── src/
    │   ├── config.py            # paths, dataset registry, seed, query-split constants
    │   ├── data_loader.py       # HDF5 inspection, loading, GloVe normalisation, query split
    │   ├── environment_info.py  # machine specs + library versions -> ../results/metadata/
    │   ├── validate_data.py     # streaming dataset validation -> ../results/metadata/
    │   ├── recall.py            # distance-based recall@10 with tie tolerance, float64 re-scoring
    │   ├── threads.py           # CPU topology, build/search thread phases (OpenMP + OpenBLAS + affinity)
    │   ├── timing.py            # single/batch timing (warm-up, median of repeats), benchmark lock
    │   ├── provenance.py        # git provenance, hypothesis guard for Parts 3/4
    │   ├── results_io.py        # results.csv schema and writer
    │   ├── indexes.py           # FAISS HNSW / IVF-Flat / IVF-PQ builders, shared seeded k-means
    │   ├── pareto.py            # Pareto front and target-recall selection
    │   └── synthetic.py         # Part 4 seeded data generator + distance statistics
    ├── tests/                   # pytest unit tests (synthetic HDF5 fixtures, no real data needed)
    ├── experiments/
    │   ├── part1_exact.py       # Part 1 driver
    │   ├── part2_ann.py         # Part 2 driver (tune -> select -> eval)
    │   ├── part2_report.py      # Part 2 figures and tables from results.csv
    │   ├── build_scaling.py     # build time with P-core threads vs 1 thread
    │   ├── part3_compression.py # Part 3 driver (hypothesis guard first)
    │   ├── part3_report.py      # Part 3 figure and tables
    │   ├── part4_dimensionality.py # Part 4 driver (hypothesis guard first)
    │   ├── part4_report.py      # Part 4 figure and tables
    │   └── probes/              # diagnostics behind design decisions (Windows)
    ├── hypothesis.md            # Part 3 / Part 4 hypotheses (author-written, committed first)
    ├── IMPLEMENTATION_LOG.md    # technical journal
    ├── run_all.sh               # one-command reproduction
    └── requirements.txt
```

## Environment setup

Python 3.12 was used. From inside `code/`:

```bash
python -m venv .venv
```

```bash
.venv/Scripts/python -m pip install -r requirements.txt
```

On Linux/macOS use `.venv/bin/python` instead. `run_all.sh` needs a bash shell. On Windows,
use Git Bash.

## Dataset setup (manual step)

Download both ann-benchmarks files (each about 0.5 GB) into `Assignment 3/data/`:

| Dataset | File | URL |
|---|---|---|
| SIFT-1M (128-d, Euclidean) | `sift-128-euclidean.hdf5` | http://ann-benchmarks.com/sift-128-euclidean.hdf5 |
| GloVe-100 (angular) | `glove-100-angular.hdf5` | http://ann-benchmarks.com/glove-100-angular.hdf5 |

To keep the data somewhere else, set `ANN_DATA_DIR` to that folder. Do not rename the files.

## Commands available

| Command (run inside `code/`) | Purpose |
|---|---|
| `.venv/Scripts/python -m src.environment_info` | write `../results/metadata/environment.json` |
| `.venv/Scripts/python -m src.validate_data` | validate datasets, write `../results/metadata/dataset_*.json` (exit 2 if missing) |
| `.venv/Scripts/python -m pytest -v` | run unit tests |
| `.venv/Scripts/python -m experiments.part1_exact --calibrate` | Part 1 on both datasets (about 12 min); rows -> `../results/results.csv` |
| `.venv/Scripts/python -m experiments.part1_exact --quick --calibrate` | Part 1 on the first 100K vectors; outputs -> `../results/quick/` |
| `.venv/Scripts/python -m experiments.probes.pinning_probe` | diagnostic: QPS pinned to each CPU (about 4 min) |
| `.venv/Scripts/python -m experiments.part2_ann` | Part 2, full grid on both datasets (about 82 min); `--dataset`/`--index` select parts |
| `.venv/Scripts/python -m experiments.part2_ann --quick` | Part 2 quick grid on the first 100K vectors (about 45 s) |
| `.venv/Scripts/python -m experiments.part2_report [--quick]` | regenerate Part 2 figures and tables from results.csv |
| `.venv/Scripts/python -m experiments.build_scaling [--quick]` | build-time scaling check (about 9 min) |
| `.venv/Scripts/python -m experiments.part3_compression [--quick] [--dev]` | Part 3 (about 22 min; quick about 2 min). Refuses to start (exit 3) unless the Part 3 hypothesis is committed and the tree is clean; `--dev` allows a dirty tree but marks rows non-reportable |
| `.venv/Scripts/python -m experiments.part3_report [--quick]` | Part 3 figure and tables from results.csv |
| `.venv/Scripts/python -m experiments.part4_dimensionality [--quick] [--dev]` | Part 4 (about 5 min; quick about 10 s); hypothesis guard first, as in Part 3 |
| `.venv/Scripts/python -m experiments.part4_report [--quick]` | Part 4 figure and tables |
| `bash run_all.sh` / `bash run_all.sh --quick` | environment, tests, validation, Parts 1-4 (quick: a few minutes; full: about 2.5 h) |

Benchmarks hold a lock file `../results/.benchmark.lock`, so two cannot run at once. Do not edit tracked files while a
benchmark is running: rows record the git state, and a dirty tree makes them non-reportable.

## Requirements checklist

Legend: [x] done and validated, [~] partially done, [ ] not started, [!] blocked

**Data**
- [x] Loader for ann-benchmarks HDF5 format (tested on synthetic files)
- [x] GloVe L2 normalisation (tested), zero-norm rows counted
- [x] 5,000 tune / 5,000 eval query split (tested)
- [x] Validation of the real SIFT and GloVe files (both PASSED, `results/metadata/dataset_*.json`)

**Part 1: exact baseline**
- [x] FAISS exact search (`IndexFlatL2`)
- [x] own recall@10: distance-based, ties at rank 10 not penalised, denominator 10, duplicates counted once
- [x] recall unit tests for tied and untied cases (16 tests)
- [x] measured count of tied queries per dataset (SIFT 137, GloVe 2 at 1e-6; sensitivity recorded)
- [x] exact vs ground-truth agreement on all 10,000 queries (recall 1.000; id differences are all ties)
- [x] single-query vs batch timing + explanation (8.6× / 8.4×; mechanism in IMPLEMENTATION_LOG.md, M2)

**Part 2: ANN benchmarks**
- [x] HNSW / IVF-Flat / IVF-PQ (all FAISS) build-parameter grids + search sweeps, both datasets (12 builds per dataset)
- [x] single-thread, one-query-at-a-time QPS, median of repeats, P-core pinning
- [x] multi-threaded builds on the verified P-core logical CPUs, with OpenMP **and** OpenBLAS thread counts set and verified, and build timing separate from query timing (design approved; controls verified)
- [x] no concurrent benchmarks (lock file, sequential stages)
- [x] recall-vs-QPS plots, QPS@0.90/0.95 table, build time / size / bytes-per-vector table
- [x] explanation of any recall ceiling (IVF-PQ plateau set by code size m; details in IMPLEMENTATION_LOG.md, M3)

**Part 3: compression (SIFT)**
- [x] hypothesis committed first (`55016fe`; the guard ran before the run, and every row records it)
- [x] PQ sub-vector sweep (m = 8, 16, 32, 64 at nlist 4096)
- [x] IVF-PQ vs IVF-PQ + exact re-rank of top 100: recall, speed, memory, candidate-pool recall, re-rank cost
- [x] tuned on the first 5,000 queries, evaluated on the last 5,000; hypotheses 3.1-3.3 evaluated in IMPLEMENTATION_LOG.md (M4)

**Part 4: dimensionality (synthetic)**
- [x] hypothesis committed first (`55016fe`; the guard ran before each run)
- [x] seeded generator (`src/synthetic.py`; byte-identical across runs, fingerprinted)
- [x] d = 32/64/128 (control)/256/512, 100K vectors, 1K queries, exact + HNSW + IVF-Flat at fixed settings
- [x] main figure (`part4_dimensionality.png`); ≤ 1-page write-up in the report

**Part 5: report and submission**
- [~] environment and version record (`results/metadata/environment.json`)
- [~] README / requirements.txt / hypothesis.md (exist, will grow)
- [x] top-level results/results.csv (Parts 1-4: 599 rows), results/figures/ (4 figures), results/tables/
- [x] run_all.sh --quick under 15 min (259 s, measured); full mode: stages run separately, about 2.2 h in total
- [x] top-level ai_log/ (prompts, responses, readable transcript, ai_statement.md)
- [x] report.pdf (5 pages, answers Q1-Q5; author-written)

**Provenance (design approved)**
- [x] hypothesis guard: Part 3/4 refuse to start if the hypothesis is missing, empty, template-only or uncommitted (used for Part 3; refused a dirty-tree run as designed)
- [x] provenance columns in every result row (pre-run commit, UTC timestamp, experiment id, config, seed, input id, metrics)
- [~] after each run, a journal summary with result-file hashes is committed (done for Part 1)

## Known limitations

- The CPU (i5-13500) is a hybrid design with 6 performance cores and 8 efficiency cores. Windows reports logical
  processors 0-11 as P-cores and 12-19 as E-cores. Timing runs will be pinned to one P-core logical processor,
  chosen at runtime, with an unpinned fallback that is recorded. Whether pinning reduces noise is not yet measured.
- `results/` and `ai_log/` live outside git (submission layout), so their history is not versioned.
- FAISS on Windows has two separate thread pools: OpenMP (`vcomp140`) and its own OpenBLAS, which defaults to 20 threads.
  Both are set explicitly. Batched exact search runs through OpenBLAS.
- Hypothesis provenance relies on git ordering plus recorded hashes. It is **not tamper-proof**, since clocks and git dates can be set
  freely. Pushing each hypothesis commit to GitHub before running adds an independent timestamp.
- `faiss.get_compile_options()` returns an empty string for this wheel, so the SIMD level is not recorded yet.

## Thread and provenance policy (summary)

| Phase | Affinity | OpenMP | OpenBLAS |
|---|---|---|---|
| index build (train + add) | all P-core logical CPUs (derived at runtime; 0-11 here) | 12 | 12 |
| query timing | one P-core logical CPU (10 here) | 1 | 1 |

Part 3/4 drivers call a hypothesis guard first. It refuses to run (exit 3) unless the relevant section of `hypothesis.md` is
filled in, committed, and identical to the committed version. Details: `IMPLEMENTATION_LOG.md`, decisions A5 and A6.
Diagnostic probes for the thread controls are in `experiments/probes/` (Windows only).

## Reproducing the results

From inside `code/`, with the datasets in `../data/` and the environment set up as above:

```bash
bash run_all.sh --quick
```

```bash
bash run_all.sh
```

`--quick` uses the first 100,000 base vectors (20,000 for Part 4) and writes to `../results/quick/`. It took 4 min 19 s on the dev desktop.
The full run writes to `../results/` and takes about 2.2 hours.
