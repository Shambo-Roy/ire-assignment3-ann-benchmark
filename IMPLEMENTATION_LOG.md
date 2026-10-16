# Implementation Log: IRE Assignment 3 (ANN benchmarking)

This is the technical record of what was built, why, how it was tested, and what the
measurements show. **Measured** results are labelled as such. Plans and expected behaviour
are labelled **PLANNED** or **EXPECTED**.

---

## 1. Assignment summary and requirements checklist

Source of truth: `../Assignment 3 Benchmarking Approximate Nearest Neighbour Search.pdf` (3 pages, read in full on 2026-10-16).
Due: 25 Oct 2026, 23:59 IST. Submission: `<rollnumber>_A3.zip` containing `report.pdf`, `code/` (this git repo with `.git`),
and `results/results.csv`, `results/figures/` and `ai_log/`.

| # | Requirement (from PDF) | Status | Evidence |
|---|---|---|---|
| D1 | SIFT-1M (128-d) and GloVe-100 (1.18M, 100-d) from ann-benchmarks | COMPLETED (validated) | `results/metadata/dataset_*.json` = PASSED |
| D2 | Normalise GloVe, Euclidean distance everywhere | COMPLETED (code + synthetic tests) | `l2_normalize`, 3 tests |
| D3 | Tune on the first 5,000 queries, report on the last 5,000 | COMPLETED (code + tests) | `split_queries`, 3 tests |
| P1.1 | Exact search with FAISS as reference | PLANNED (M2) | |
| P1.2 | Own recall@10 that does not count ties at rank 10 as mistakes, plus unit tests | PLANNED (M2) | |
| P1.3 | Exact results agree with the stored neighbours | PLANNED (M2). M1 spot check is ready but has not run on real data | |
| P1.4 | Time exact search one query at a time vs batched, and explain the difference | PLANNED (M2) | |
| P2.1 | HNSW, IVF-Flat, IVF-PQ on both datasets: several build settings, search-parameter sweep up to recall ≥ 0.95 | PLANNED (M3) | |
| P2.2 | QPS measured one query at a time, single thread, median of repeats | PLANNED (M2 harness, M3 use) | |
| P2.3 | Recall-vs-QPS plot per dataset (Pareto front per index + exact) | PLANNED | |
| P2.4 | Table: QPS at recall 0.90 / 0.95 | PLANNED | |
| P2.5 | Table: build time, index size, bytes/vector at recall 0.95 | PLANNED | |
| P2.6 | Explain any recall ceiling | PLANNED | |
| P3 | Compression on SIFT: vary PQ m; IVF-PQ vs IVF-PQ + exact rerank of top 100; recall, speed, memory; hypothesis committed first | PLANNED | |
| P4 | Own synthetic data: dimensionality d ∈ {32, 64, 128 (control), 256, 512}, ≤ 500K vectors, seeded, ≥ 2 indexes, hypothesis committed first, 1 figure, ≤ 1 page | PLANNED | |
| P5.1 | Report ≤ 5 pages that answers Q1–Q5 with mechanisms | PLANNED | |
| P5.2 | Machine specs + library versions | IN PROGRESS | `results/metadata/environment.json` |
| P5.3 | `README.md`, `requirements.txt`, `hypothesis.md` | IN PROGRESS | files exist |
| P5.4 | `bash run_all.sh` (full) and `--quick` (under 15 min on a laptop) | IN PROGRESS (stub) | `run_all.sh` |
| P5.5 | `results/results.csv` (one row per run behind any reported number), `results/figures/` | PLANNED | |
| P5.6 | `ai_log/` with all prompts and responses, plus `ai_statement.md` classifying every file | IN PROGRESS | `ai_log/` |
| P5.7 | No datasets or index files in the zip; no ann-benchmarks framework; no copied numbers | ENFORCED | `.gitignore`, data dir outside repo |
| P5.8 | Own recall, timing and plotting code (FAISS/hnswlib only for indexes) | POLICY | |

---

## 2. Environment and machine specifications

**Measured** on 2026-10-16 by `python -m src.environment_info` (full record: `results/metadata/environment.json`):

| Item | Value |
|---|---|
| Machine | HP Pro Tower 400 G9 PCI Desktop PC (from `Win32_ComputerSystem`) |
| CPU | 13th Gen Intel Core i5-13500, 14 physical cores / 20 logical |
| CPU max freq (as reported by Windows) | 2500 MHz. This is the base clock; Windows does not report turbo |
| RAM | 15.68 GB total, ~7.0 GB available at collection time |
| Disk | 835 GB free on C: (`df -h`) |
| OS | Windows 11 Pro 10.0.26200 |
| Python | 3.12.10 (CPython, MSC v.1943, 64-bit), project venv `code/.venv` |
| numpy / faiss-cpu / h5py / psutil / pytest | 2.5.4 / 1.15.1 / 3.16.0 / 7.2.2 / 9.1.1 |
| FAISS default OpenMP threads | 20 (so benchmarks **must** call `faiss.omp_set_num_threads(1)`) |
| `faiss.get_compile_options()` | empty string. The SIMD level is not yet known |
| Not installed | matplotlib, pandas, hnswlib (in the venv) |

**Note:** the global Python install also has numpy 2.5.2, matplotlib and others, but all project
code runs from the isolated venv so those versions are not mixed in.

**Measurement risk:** the i5-13500 is a hybrid CPU with 6 performance cores and 8 efficiency cores,
which matches 14 cores / 20 threads. A single-threaded benchmark may run on either core type, and that changes QPS. Mitigation for M2: pin
the benchmark process to one P-core with an affinity mask, or at least run every configuration under the same
conditions, then record the choice.

---

## 3. Dataset locations, formats, dimensions and validation results

**Status: COMPLETED. Both files PASSED all checks** (2026-10-16, `python -m src.validate_data`, exit 0, 2.7 s wall time).
The author downloaded both files into `Assignment 3/data/`. Full reports: `results/metadata/dataset_{sift,glove}.json`.

**Measured validation results**

| | SIFT | GloVe |
|---|---|---|
| File size | 525,128,288 B | 485,413,888 B |
| `distance` attribute | euclidean | angular |
| train | 1,000,000 × 128 float32 | 1,183,514 × 100 float32 |
| test (queries) | 10,000 × 128 float32 | 10,000 × 100 float32 |
| neighbors / distances | 10,000 × 100 int32 / float32 | 10,000 × 100 int32 / float32 |
| HDF5 chunking / compression | none / none (contiguous) | none / none |
| NaN / inf (train, queries) | 0 / 0 | 0 / 0 |
| Zero-norm rows (train, queries) | 0, 0 | 0, 0 |
| Raw train L2 norm range | 506.3 – 511.4 | 0.041 – 18.68 |
| Raw query L2 norm range | 506.7 – 511.1 | 1.11 – 17.09 |
| GT ids range | 5 … 999,998 (in range) | 0 … 1,183,513 (in range) |
| GT distance rows not sorted | 0 | 0 |
| **Exact ties: distance(rank 10) == distance(rank 11)** | **137 queries** | **1 query** |
| Spot check (10 queries × 100 neighbours): max abs error vs stored | 0.0 (stored = L2) | 1.5e-7 (stored = ‖q̂−x̂‖²/2 = 1 − cos) |
| Ranking preserved in our metric (spot check) | 10/10 | 10/10 |
| Normalised 10K-row sample: max abs deviation of norm from 1 | n/a | 2.4e-7 |
| 5,000 / 5,000 split | OK | OK |

**Interpretation**
- *SIFT norms of about 512:* the descriptors in this release appear to be scaled to almost constant norm, with integer components.
  Integer components give integer squared distances, which explains the 137 exact rank-10/11 ties. This shows **tie handling
  in recall@10 (Part 1) is not hypothetical**: up to 1.4% of SIFT queries could be penalised wrongly by a naive id-matching recall.
- *GloVe:* the stored distances are confirmed to be cosine distance (1 − cos), so normalisation plus Euclidean reproduces the
  stored ranking (error 1.5e-7, float32 round-off). No zero vectors exist, so the zero-norm handling is never triggered on this data.
- *Not yet established:* that **full** exact search reproduces the stored top-10 for all 10,000 queries. That is Part 1 (M2).
  The spot check covers only 10 queries.

| Dataset | Expected path (default) | URL | Expected contents (from the spec, to be verified) |
|---|---|---|---|
| SIFT | `Assignment 3/data/sift-128-euclidean.hdf5` | http://ann-benchmarks.com/sift-128-euclidean.hdf5 | train 1,000,000×128 f32; test 10,000×128; neighbors/distances 10,000×100; attr `distance=euclidean` |
| GloVe | `Assignment 3/data/glove-100-angular.hdf5` | http://ann-benchmarks.com/glove-100-angular.hdf5 | train 1,183,514×100 f32; test 10,000×100; neighbors/distances 10,000×100; attr `distance=angular` |

Format (ann-benchmarks HDF5): datasets `train`, `test`, `neighbors`, `distances`, plus a file attribute `distance`.

(Before the download, `validate_data` correctly reported MISSING with exit code 2.)

---

## 4. Architecture and source-file responsibilities

```
Assignment 3/
├── Assignment 3 ... .pdf    spec (outside repo)
├── data/                    datasets (outside repo, configurable via ANN_DATA_DIR)
└── code/                    git repo = submission "code/"
    src/config.py            single place for paths, dataset registry, SEED, split sizes, K
    src/data_loader.py       inspect_hdf5, chunked_stats, l2_normalize, split_queries, load_dataset
    src/environment_info.py  collect() -> results/metadata/environment.json
    src/validate_data.py     validate(spec) -> results/metadata/dataset_<name>.json
    tests/conftest.py        writes small ann-benchmarks-format HDF5 files with exact ground truth
    tests/test_*.py          unit tests
    run_all.sh               reproduction entry point (stub)
```

Data flow (planned): `config` → `data_loader.load_dataset` → index builders (M3) → timing harness (M2)
→ `results/results.csv` → plotting/table scripts → `results/figures/`.

**Why the data directory is outside `code/`:** the zip must not contain datasets. Keeping the data
outside the repo makes that impossible to get wrong, and `.gitignore` also excludes `*.hdf5` and index files.

---

## 5. Milestone history

### Milestone 1: Environment and data validation (2026-10-16)

**Status:** COMPLETED. Code and tests were done first; real-data validation PASSED on 2026-10-16 after the datasets were downloaded.

**Files created:** `src/__init__.py`, `src/config.py`, `src/data_loader.py`, `src/environment_info.py`,
`src/validate_data.py`, `tests/__init__.py`, `tests/conftest.py`, `tests/test_data_loader.py`,
`tests/test_environment_info.py`, `pytest.ini`, `requirements.txt`, `.gitignore`, `run_all.sh`,
`hypothesis.md` (template only), `README.md`, `IMPLEMENTATION_LOG.md`, `ai_log/ai_statement.md`,
`ai_log/milestone1/prompt.md`, `ai_log/milestone1/response.md`. Generated: `results/metadata/environment.json`,
`results/metadata/dataset_sift.json`, `results/metadata/dataset_glove.json`.
No pre-existing files were modified. The workspace contained only the PDF.

**What was implemented**
1. Environment recorder (`environment_info.py`). It records Python, OS, CPU model (via the OS, since `platform.processor()`
   is vague), core counts, RAM, package versions and FAISS's default thread count, and writes JSON.
2. Metadata-only inspection (`inspect_hdf5`). It reads shapes, dtypes, chunking and attributes from the HDF5 headers, without loading any data.
3. Streaming statistics (`chunked_stats`). It counts NaN, inf and zero-norm rows and records min/max norm, in blocks of 100K rows,
   so peak memory is about 51 MB for SIFT instead of 512 MB.
4. Validation (`validate_data.py`). It checks required keys, shapes and dtypes against the registry, the `distance` attribute,
   that ground-truth ids are in range, and that ground-truth distances are sorted per row. It also counts exact ties between
   ranks 10 and 11, and runs a **spot check** that recomputes the distances to the 100 stored neighbours for 10 seeded random queries.
   For GloVe it also checks that normalisation preserves ranking on those queries and produces unit norms on a 10K-row sample. The split is checked to be 5,000/5,000.
5. GloVe normalisation (`l2_normalize`). It returns a float32 copy with unit-norm rows. Zero rows stay zero and are counted.
6. Query split (`split_queries`). It uses contiguous slices, `[0, 5000)` for tuning and `[5000, 10000)` for evaluation. There is no shuffling.
7. `load_dataset` loads the full arrays (planned for M2+). It normalises only if `spec.normalize` is set and exposes
   `tune_queries` / `eval_queries` / `tune_gt` / `eval_gt`.

**Key decisions**
- *`code/` as the git root.* This matches the required zip layout directly.
- *Project venv.* FAISS was missing from the global Python. Installing it into `code/.venv` leaves the
  global environment unchanged and records the exact versions.
- *Why normalisation preserves the GloVe ground truth.* For unit vectors, ‖q−x‖² = 2 − 2·cos(q,x). Euclidean
  ranking after normalisation is therefore identical to cosine ranking, which is the metric ann-benchmarks used
  for the stored neighbours. The stored GloVe `distances` are expected to be cosine distance (1 − cos), so
  our ‖q−x‖²/2 should reproduce them. The spot check verifies this on real data rather than assuming it.
  This is tested in `test_normalised_euclidean_ranking_equals_cosine_ranking`.
- *Zero-norm vectors.* They are kept as zero vectors, not dropped, so ids stay aligned with the ground truth, and they are counted in the report.
  A zero vector has distance ‖q‖² = 1 to every unit query, which is far from everything, so it is harmless.
- *Contiguous split, no shuffle.* The assignment says "first 5,000" and "last 5,000" literally.
- *Folder name `ai_log/` and script name `run_all.sh`.* The PDF renders LaTeX underscores as spaces ("ai log", "run all.sh").
  The intended names have underscores, consistent with `run_all.sh` in the prompt.
- *Synthetic HDF5 fixtures for tests.* Tests must not depend on 1 GB of downloads. The fixtures use the identical
  layout with brute-force ground truth. They keep 10,000 queries so the real split sizes are exercised.

**Commands executed** (from `Assignment 3/` or `code/`)
```
ls -la; find . -maxdepth 3; git status                     # workspace inspection (no git repo found)
python --version; pip list; df -h; Get-CimInstance ...     # environment inspection
find ~/Downloads ~/Desktop ~/Documents ... -iname '*sift*' # dataset search (none found)
mkdir code data; cd code; git init -b main
python -m venv .venv
.venv/Scripts/python -m pip install --upgrade pip
.venv/Scripts/python -m pip install numpy h5py faiss-cpu psutil pytest
.venv/Scripts/python -c "import faiss; ... IndexFlatL2 smoke test"   # OK, returned self as NN
.venv/Scripts/python -m pytest -v
.venv/Scripts/python -m src.environment_info
.venv/Scripts/python -m src.validate_data                 # exit 2 (MISSING x2)
bash run_all.sh --quick                                   # tests pass, then exit 2 at validation (expected)
```

**Tests (measured outcome): 18 passed, 0 failed, in about 2.2 s** (`pytest -v`, 2026-10-16). See section 10.

**What this establishes:** the loader, validator, normaliser and split behave correctly on files in the
ann-benchmarks layout, including detecting NaN, wrong shape, wrong metric attribute and missing keys.
**What it does not establish:** that the real files match the expected format, or that the real GloVe stored distances follow
the cosine-distance relation. Both need the datasets.

**Update 2026-10-16 (real-data validation):** the author downloaded both datasets. `python -m src.validate_data` → both
PASSED, exit 0 (see section 3). **Milestone 1 status: COMPLETED.**

**Remaining / next:** Milestone 2 (Part 1 exact baseline), after the author's review.

---

## 6. Algorithms, formulas and implementation decisions

- L2 normalisation: x̂ = x / ‖x‖₂ (rows with ‖x‖ = 0 stay 0).
- Equivalence: ‖q̂ − x̂‖² = 2 − 2 cos(q, x), so ranking is identical to angular ranking.
- **PLANNED (M2), recall@10 with ties:** for query q, let r₁₀ be the true distance of the 10th nearest neighbour.
  A returned id counts as correct if its true distance to q is ≤ r₁₀ + ε. This way, a vector tied with the 10th
  neighbour is not counted as a mistake even if its id differs from the stored one. Recall = (#correct among
  the returned top 10) / 10. The exact definition and ε will be fixed and unit-tested in M2.

## 7. Hypotheses and commit identifiers

| Experiment | Hypothesis commit | Results commit | Status |
|---|---|---|---|
| Part 3: compression | not yet | not yet | template in `hypothesis.md`; the **author must write** it |
| Part 4: dimensionality | not yet | not yet | template in `hypothesis.md`; the **author must write** it |

## 8. Configuration parameters

| Parameter | Value | Reason |
|---|---|---|
| `SEED` | 12345 | single global seed (spot-check sampling now; k-means sampling and synthetic data later) |
| `N_TUNE_QUERIES` | 5000 | assignment |
| `K` | 10 | recall@10 |
| `chunk_rows` | 100,000 | keeps streaming validation under ~60 MB for these dimensionalities |
| `N_SPOT_CHECK` | 10 queries | cheap ground-truth sanity check. The full check is Part 1 |
| `ANN_DATA_DIR` | default `../data` | portability |

## 9. Commands

See the Milestone 1 entry. User-facing commands are listed in `README.md`.

## 10. Tests and validation results

Milestone 1, **measured** 2026-10-16: `18 passed in 2.19s`.

| Test | What it verifies |
|---|---|
| test_inspect_reports_shapes_and_attrs | header-only metadata is correct |
| test_inspect_flags_missing_keys | malformed files are detected |
| test_registry_paths_are_relative_to_repo_not_machine | `ANN_DATA_DIR` override works, no hardcoded paths |
| test_validate_passes_on_wellformed_euclidean | the validator accepts a correct SIFT-like file |
| test_validate_passes_on_wellformed_angular_with_normalisation | the validator accepts a correct GloVe-like file. Spot-check error < 1e-4 confirms the sq-L2/2 = cosine-distance relation |
| test_validate_reports_missing_file | MISSING status |
| test_validate_detects_nan_and_wrong_shape | NaN injected at train[5,3] is caught; wrong dim is caught |
| test_validate_detects_wrong_metric_attribute | metric mismatch is caught |
| test_split_is_5000_5000_disjoint_and_ordered | exact split boundaries |
| test_split_rejects_degenerate ×4 | invalid split sizes raise |
| test_loaded_dataset_exposes_matching_query_and_gt_halves | query rows and GT rows stay aligned in each half |
| test_l2_normalize_unit_rows_and_zero_rows_counted | unit norms, zero row stays zero (no NaN), input not mutated |
| test_normalised_euclidean_ranking_equals_cosine_ranking | the equivalence formula, numerically |
| test_load_dataset_normalises_only_when_spec_says_so | normalisation applied only to GloVe, file on disk unchanged |
| test_collect_has_required_fields_and_is_json_serialisable | environment record |

Real-data validation, **measured** 2026-10-16: SIFT PASSED, GloVe PASSED (all checks; details in section 3).

## 11. Bugs encountered

None in Milestone 1.

## 12. Benchmark measurements

None yet. No benchmarks have been run.

## 13. Figures and tables

None yet.

## 14. Limitations, assumptions, unresolved issues

- The expected GloVe train count (1,183,514) and the cosine-distance encoding of stored distances come from the ann-benchmarks conventions. The validator will check them.
- Hybrid P/E-core CPU: single-thread timing noise (see section 2).
- Only ~7 GB of RAM was free at inspection time. Loading one dataset (~0.5 GB) plus an index is fine. Plan to process one dataset at a time.
- `run_all.sh` is a stub. `--quick` needs a subset (for example the first 100K train vectors). On a subset the stored ground truth is
  **invalid**, so it must be recomputed with exact FAISS search. This is planned for M2.
- The global Python environment is not used. Graders should install from `requirements.txt`.
- The SIMD level of the FAISS wheel is not recorded yet.

## 15. Decisions / actions required from the user

1. ~~Download the datasets~~. Done 2026-10-16, both validated.
2. **Confirm the layout:** is `code/` as the git root and `data/` as its sibling acceptable?
3. **Decide where `results/` and `ai_log/` go in the zip.** The PDF lists them as separate deliverables. Right now they live inside
   `code/` so they are versioned and produced by `run_all.sh`. They can be copied to the zip root at submission time if wanted.
4. **Make the first git commit** (not done automatically). See README / the final response for the command.
5. **Save the session transcript** into `ai_log/transcripts/` (see `ai_log/ai_statement.md`).
6. **P-core pinning** for timing in M2: approve or decline.
7. **hnswlib:** the assignment allows it, but the plan uses FAISS `IndexHNSWFlat` only, for one consistent library. Confirm.

## 16. Report preparation map

| Report item | Evidence needed | Source (planned) |
|---|---|---|
| Q1 fastest index at recall 0.95, per dataset | QPS@0.95 table + Pareto plots | M3 results.csv, figures |
| Q2 build time / memory; what PQ saves and loses | build/size table + Part 3 | M3, M4 |
| Q3 Part 3/4 findings; when headline numbers mislead | hypothesis vs results, Part 4 figure | M4, M5 |
| Q4 150K vs 10M vectors, 100 ms, 16 GB | exact single-query latency (Part 1) and its scaling with n, index bytes/vector | M2, M3 |
| Q5 two ways the benchmark could mislead | e.g. P/E-core scheduling, tune/eval split, single-thread vs batch, warm cache | all |
| Machine specs / versions | `results/metadata/environment.json` | M1 ✔ |
