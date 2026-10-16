# IRE Assignment 3: Benchmarking Approximate Nearest Neighbour Search

CS4.406 Information Retrieval & Extraction. This repo benchmarks exact search, HNSW, IVF-Flat and
IVF-PQ (FAISS) on SIFT-1M and GloVe-100, then runs a compression experiment (Part 3) and a
synthetic dimensionality experiment (Part 4).

## Current status

**Milestone 1 (environment + data validation): COMPLETED.** Both datasets are downloaded and pass validation.
Next is Milestone 2 (Part 1). See `IMPLEMENTATION_LOG.md` for the full record.

## Directory layout

```
Assignment 3/
├── data/                     # datasets live here, OUTSIDE the git repo (never committed/zipped)
└── code/                     # this git repository = the "code/" folder of the submission
    ├── src/
    │   ├── config.py         # paths, dataset registry, seed, query-split constants
    │   ├── data_loader.py    # HDF5 inspection, loading, GloVe normalisation, query split
    │   ├── environment_info.py  # machine specs + library versions -> results/metadata/
    │   └── validate_data.py  # streaming dataset validation -> results/metadata/
    ├── tests/                # pytest unit tests (synthetic HDF5 fixtures, no real data needed)
    ├── experiments/          # (planned) benchmark drivers for Parts 1-4
    ├── results/
    │   ├── metadata/         # environment.json, dataset_<name>.json
    │   ├── results.csv       # (planned) one row per benchmark run
    │   └── figures/          # (planned)
    ├── ai_log/               # AI prompts/responses + ai_statement.md
    ├── hypothesis.md         # Part 3 / Part 4 hypotheses (author-written, committed first)
    ├── IMPLEMENTATION_LOG.md # technical journal
    ├── run_all.sh            # one-command reproduction (partial)
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
| `.venv/Scripts/python -m src.environment_info` | write `results/metadata/environment.json` |
| `.venv/Scripts/python -m src.validate_data` | validate datasets, write `results/metadata/dataset_*.json` (exit 2 if missing) |
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
- [ ] FAISS exact search, [ ] own recall@10 with tie handling, [ ] recall unit tests
- [ ] exact vs ground-truth agreement, [ ] single-query vs batch timing + explanation

**Part 2: ANN benchmarks**
- [ ] HNSW / IVF-Flat / IVF-PQ build-parameter grids + search sweeps, both datasets
- [ ] single-thread, one-query-at-a-time QPS, median of repeats
- [ ] recall-vs-QPS plots, QPS@0.90/0.95 table, build time / size / bytes-per-vector table
- [ ] explanation of any recall ceiling

**Part 3: compression (SIFT)**
- [ ] hypothesis committed first, [ ] PQ sub-vector sweep, [ ] IVF-PQ vs IVF-PQ + rerank top-100

**Part 4: dimensionality (synthetic)**
- [ ] hypothesis committed first, [ ] seeded generator, [ ] d = 32/64/128(control)/256/512, exact + HNSW + IVF-Flat, [ ] main figure

**Part 5: report and submission**
- [~] environment and version record (`results/metadata/environment.json`)
- [~] README / requirements.txt / hypothesis.md (exist, will grow)
- [ ] results/results.csv, results/figures/
- [~] run_all.sh (+ --quick under 15 min)
- [~] ai_log/ (Milestone 1 prompt and response; the author still has to copy in the transcript)
- [ ] report.pdf (max 5 pages, answers Q1-Q5)

## Known limitations

- The CPU (i5-13500) is a hybrid design with 6 performance cores and 8 efficiency cores. Windows
  may put a single-thread benchmark on either core type, which adds timing noise. Mitigation is planned for Milestone 2.
- `faiss.get_compile_options()` returns an empty string for this wheel, so the SIMD level is not recorded yet.

## Next milestone

Milestone 2 covers the Part 1 exact baseline. It adds a recall@10 function with tie handling and
its unit tests, FAISS exact search, a check against the stored ground truth, a timing harness and
the start of `results/results.csv`.
