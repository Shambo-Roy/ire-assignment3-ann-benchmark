# IRE Assignment 3: Benchmarking Approximate Nearest Neighbour Search

CS4.406 Information Retrieval & Extraction. This repo benchmarks exact search, HNSW, IVF-Flat and
IVF-PQ (FAISS) on SIFT-1M and GloVe-100, then runs a compression experiment (Part 3) and a
synthetic dimensionality experiment (Part 4).

## Current status

**Milestone 1 (environment + data validation): COMPLETED.** Both datasets are downloaded and pass validation.
Next is Milestone 2 (Part 1). See `IMPLEMENTATION_LOG.md` for the full record.

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
    │   └── validate_data.py     # streaming dataset validation -> ../results/metadata/
    ├── tests/                   # pytest unit tests (synthetic HDF5 fixtures, no real data needed)
    ├── experiments/             # (planned) benchmark drivers for Parts 1-4
    ├── hypothesis.md            # Part 3 / Part 4 hypotheses (author-written, committed first)
    ├── IMPLEMENTATION_LOG.md    # technical journal
    ├── run_all.sh               # one-command reproduction (partial)
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
| `bash run_all.sh` / `bash run_all.sh --quick` | currently: environment + tests + validation only |

## Requirements checklist

Legend: [x] done and validated, [~] partially done, [ ] not started, [!] blocked

**Data**
- [x] Loader for ann-benchmarks HDF5 format (tested on synthetic files)
- [x] GloVe L2 normalisation (tested), zero-norm rows counted
- [x] 5,000 tune / 5,000 eval query split (tested)
- [x] Validation of the real SIFT and GloVe files (both PASSED, `results/metadata/dataset_*.json`)

**Part 1: exact baseline**
- [ ] FAISS exact search (`IndexFlatL2`)
- [ ] own recall@10: distance-based, ties at rank 10 not penalised, denominator 10, duplicates counted once (design approved)
- [ ] recall unit tests for tied and untied cases
- [ ] measured count of tied queries per dataset (not hardcoded)
- [ ] exact vs ground-truth agreement on all 10,000 queries
- [ ] single-query vs batch timing + explanation

**Part 2: ANN benchmarks**
- [ ] HNSW / IVF-Flat / IVF-PQ (all FAISS) build-parameter grids + search sweeps, both datasets
- [ ] single-thread, one-query-at-a-time QPS, median of repeats, P-core pinning with verified mapping (design approved; mapping verified)
- [ ] multi-threaded builds on the verified P-core logical CPUs, with OpenMP **and** OpenBLAS thread counts set and verified, and build timing separate from query timing (design approved; controls verified)
- [ ] no concurrent benchmarks (lock file, sequential stages)
- [ ] recall-vs-QPS plots, QPS@0.90/0.95 table, build time / size / bytes-per-vector table
- [ ] explanation of any recall ceiling

**Part 3: compression (SIFT)**
- [ ] hypothesis committed first (enforced by the guard), [ ] PQ sub-vector sweep, [ ] IVF-PQ vs IVF-PQ + rerank top-100

**Part 4: dimensionality (synthetic)**
- [ ] hypothesis committed first (enforced by the guard), [ ] seeded generator, [ ] d = 32/64/128(control)/256/512, exact + HNSW + IVF-Flat, [ ] main figure

**Part 5: report and submission**
- [~] environment and version record (`results/metadata/environment.json`)
- [~] README / requirements.txt / hypothesis.md (exist, will grow)
- [ ] top-level results/results.csv, results/figures/ (output location implemented and tested)
- [~] run_all.sh (+ --quick under 15 min)
- [~] top-level ai_log/ (Milestone 1 prompt and response; the author still has to copy in the transcript)
- [ ] report.pdf (max 5 pages, answers Q1-Q5)

**Provenance (design approved)**
- [ ] hypothesis guard: Part 3/4 refuse to start if the hypothesis is missing, empty, template-only or uncommitted
- [ ] provenance columns in every result row (pre-run commit, UTC timestamp, experiment id, config, seed, input id, metrics)
- [ ] after each run, a journal summary with result-file hashes is committed

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

## Next milestone

Milestone 2 covers the Part 1 exact baseline. It adds the distance-based recall@10 and its unit tests, FAISS
exact search, a check against the stored ground truth on all queries, the pinned single-thread timing harness
(with a pinned/unpinned/E-core calibration), single-query vs batched timing, and the `results.csv` schema.
The full specification is in `IMPLEMENTATION_LOG.md`, section 6 (decisions A1-A4).
