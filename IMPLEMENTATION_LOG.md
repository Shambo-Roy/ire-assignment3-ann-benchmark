# Implementation Log: IRE Assignment 3 (ANN benchmarking)

This is the technical record of what was built, why, how it was tested, and what the
measurements show. **Measured** results are labelled as such. Plans and expected behaviour
are labelled **PLANNED** or **EXPECTED**.

---

## 1. Assignment summary and requirements checklist

Source of truth: `../Assignment 3 Benchmarking Approximate Nearest Neighbour Search.pdf` (3 pages, read in full on 2026-10-16).
Due: 25 Oct 2026, 23:59 IST. Submission: `<rollnumber>_A3.zip` with four top-level items (approved layout, decision A4 in section 6):
`report.pdf`, `code/` (this git repo with `.git`), `results/` (`results.csv`, `figures/`, `metadata/`) and `ai_log/`.

| # | Requirement (from PDF) | Status | Evidence |
|---|---|---|---|
| D1 | SIFT-1M (128-d) and GloVe-100 (1.18M, 100-d) from ann-benchmarks | COMPLETED (validated) | `results/metadata/dataset_*.json` = PASSED |
| D2 | Normalise GloVe, Euclidean distance everywhere | COMPLETED (code + synthetic tests) | `l2_normalize`, 3 tests |
| D3 | Tune on the first 5,000 queries, report on the last 5,000 | COMPLETED (code + tests) | `split_queries`, 3 tests |
| P1.1 | Exact search with FAISS as reference (`IndexFlatL2`, decision A3) | COMPLETED (M2) | `experiments/part1_exact.py`; reference cached in `data/cache/` |
| P1.2 | Own recall@10, distance-based so ties at rank 10 are not mistakes (decision A2), plus unit tests for tied and untied cases | COMPLETED (M2) | `src/recall.py`; 16 tests in `tests/test_recall.py` |
| P1.3 | Exact results agree with the stored neighbours (all 10,000 queries) | COMPLETED (M2): recall 1.000 both ways on both datasets; 65 SIFT / 1 GloVe id differences, all ties | `results/metadata/part1_*_reference.json` |
| P1.4 | Time exact search one query at a time vs batched, and explain the difference | COMPLETED (M2): 8.57× SIFT, 8.36× GloVe (1 thread, BLAS path); mechanism in the M2 entry | `results/results.csv` (`p1_exact`) |
| P1.5 | Measure and report the number of tied queries at the rank-10 boundary, not hardcoded | COMPLETED (M2): SIFT 137, GloVe 2 at 1e-6, with sensitivity table | `part1_*_reference.json` → `boundary_ties` |
| T1 | Timing method: single thread, repeated, median, P-core pinning with verified mapping (decision A1) | COMPLETED (M2); pinning validated by calibration + diagnostic | `src/threads.py`, `src/timing.py`, `pinning_probe.json` |
| T2 | Build: multi-threaded on the verified P-core logical processors. OpenMP and OpenBLAS both set and verified. Build timing kept separate from query timing. Build thread configuration recorded (A5) | COMPLETED (M3): all ANN builds under `build_phase()`; scaling check SIFT HNSW 5.45×, IVF-Flat 1.76×, IVF-PQ 1.75× | `p2_build_scaling` rows |
| T3 | No concurrent benchmarks (lock file, sequential `run_all.sh`) (A5) | COMPLETED (M2) | `benchmark_lock`, 2 tests |
| G1 | Hypothesis guard: refuse Part 3/4 if the hypothesis is missing, empty, template-only or uncommitted (A6) | IMPLEMENTED + TESTED (10 tests incl. numbered headings); accepts committed `55016fe`; used from M4 | `src/provenance.py` |
| G2 | Provenance columns in every result row: commit, UTC time, experiment id, configuration, seed, input id, metrics (A6) | COMPLETED (M2 schema, used by Part 1) | `src/results_io.py` |
| G3 | After each run: journal summary with output hashes, then commit (A6) | IN USE (M2 entry has hashes) | section 5, M2 |
| P2.1 | HNSW, IVF-Flat, IVF-PQ on both datasets: several build settings, search-parameter sweep up to recall ≥ 0.95 | COMPLETED (M3): 3 + 3 + 6 builds per dataset, 252 tune rows | `experiments/part2_ann.py`, `p2_ann` rows |
| P2.2 | QPS measured one query at a time, single thread, median of repeats | COMPLETED (M3): every row single-query, 1 thread, pinned, median (eval R = 5) | `src/timing.py` |
| P2.3 | Recall-vs-QPS plot per dataset (Pareto front per index + exact) | COMPLETED (M3) | `results/figures/part2_{sift,glove}_recall_vs_qps.png` |
| P2.4 | Table: QPS at recall 0.90 / 0.95 | COMPLETED (M3) | `results/tables/part2_qps_at_recall.md` |
| P2.5 | Table: build time, index size, bytes/vector at recall 0.95 | COMPLETED (M3) | `results/tables/part2_build_cost_at_095.md` |
| P2.6 | Explain any recall ceiling | COMPLETED (M3): IVF-PQ plateau set by m (quantization error); GloVe HNSW/IVF limits are sweep-limited | M3 entry, `part2_max_recall.md` |
| P3 | Compression on SIFT: vary PQ m; IVF-PQ vs IVF-PQ + exact rerank of top 100; recall, speed, memory; hypothesis committed first | COMPLETED (M4): m ∈ {8,16,32,64}, nlist 4096; hypotheses 3.1–3.3 supported (memory caveat on 3.3) | `p3_compression` rows (158), `part3_compression.png`, `part3_*.md` |
| P4 | Own synthetic data: dimensionality d ∈ {32, 64, 128 (control), 256, 512}, ≤ 500K vectors, seeded, ≥ 2 indexes, hypothesis committed first, 1 figure, ≤ 1 page | COMPLETED (M5) except the ≤ 1-page write-up (M6): 100K vectors × 5 settings, seeded and fingerprinted, Flat + HNSW + IVF-Flat, guard `55016fe`, main figure | `p4_dim` rows (55), `part4_dimensionality.png`, `part4_*.md`, `src/synthetic.py` |
| P5.1 | Report ≤ 5 pages that answers Q1–Q5 with mechanisms | PLANNED | |
| P5.2 | Machine specs + library versions | IN PROGRESS | `results/metadata/environment.json` |
| P5.3 | `README.md`, `requirements.txt`, `hypothesis.md` | IN PROGRESS: hypotheses written by the author and committed (`55016fe`) | files exist |
| P5.4 | `bash run_all.sh` (full) and `--quick` (under 15 min on a laptop), run from `code/` and writing to the top-level `results/` | IN PROGRESS: all experiment stages (Parts 1–4). Quick: about 4 min in total (not yet timed end to end) | `run_all.sh` |
| P5.5 | `results/results.csv` (one row per run behind any reported number), `results/figures/`, at the top level | IN PROGRESS: 599 rows (Parts 1–4), 4 figures, 8 tables | `../results/` |
| P5.6 | `ai_log/` (top level) with all prompts and responses, plus `ai_statement.md` classifying every file | IN PROGRESS | `../ai_log/` |
| P5.7 | No datasets or index files in the zip; no ann-benchmarks framework; no copied numbers | ENFORCED | `.gitignore`, data dir outside repo |
| P5.8 | Own recall, timing and plotting code; FAISS for all indexes (decision A3) | POLICY | |
| P5.9 | Git history shows each hypothesis committed before its results | PLANNED | mechanism in section 7 (results are not in git) |

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
| FAISS default OpenMP threads | 20 (benchmarks **must** call `faiss.omp_set_num_threads(1)`) |
| FAISS bundled OpenBLAS | 0.3.30, own thread pool, **default 20 threads**, `NO_AFFINITY` (design revision B). It must be set separately |
| `faiss.get_compile_options()` | empty string. The SIMD level is not yet known |
| Not installed | matplotlib, pandas, hnswlib (in the venv) |

**Note:** the global Python install also has numpy 2.5.2, matplotlib and others, but all project
code runs from the isolated venv so those versions are not mixed in.

**Measurement risk:** the i5-13500 is a hybrid CPU with 6 performance cores and 8 efficiency cores,
which matches 14 cores / 20 threads. A single-threaded benchmark may run on either core type, and that changes QPS.

**Logical-processor mapping, measured 2026-10-16.** Probe: Windows `GetSystemCpuSetInformation` via ctypes,
read-only, scratchpad script, not yet in the repo.

| Logical processors | Physical core index | EfficiencyClass | Type |
|---|---|---|---|
| 0,1 · 2,3 · 4,5 · 6,7 · 8,9 · 10,11 | 0, 2, 4, 6, 8, 10 (two SMT threads each) | 1 | P-core (6) |
| 12 … 19 | 12 … 19 (one thread each) | 0 | E-core (8) |

A higher EfficiencyClass means a faster core. The counts (6 P × 2 + 8 E × 1 = 20) match the CPU specification.
Setting the affinity with `psutil.Process().cpu_affinity([0])` worked (read back `[0]`). Pinning is therefore
**available**. Its effect on timing stability has **not yet been measured**; that is part of M2.

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
Assignment 3/                    = the submission root (zip contents, minus data/ and the PDF)
├── Assignment 3 ... .pdf        spec (not submitted)
├── data/                        datasets (not submitted; ANN_DATA_DIR overrides)
├── report.pdf                   (planned)
├── results/                     SUBMITTED top level; written by code (ANN_RESULTS_DIR overrides)
│   ├── metadata/                environment.json, dataset_<name>.json
│   ├── results.csv              (planned) one row per benchmark run
│   └── figures/                 (planned)
├── ai_log/                      SUBMITTED top level; ai_statement.md, milestone prompts/responses, transcripts/
└── code/                        SUBMITTED; git repo
    src/config.py                paths (REPO_ROOT, RESULTS_DIR = REPO_ROOT/../results), dataset registry, SEED, split, K
    src/data_loader.py           inspect_hdf5, chunked_stats, l2_normalize, split_queries, load_dataset
    src/environment_info.py      collect() -> ../results/metadata/environment.json
    src/validate_data.py         validate(spec) -> ../results/metadata/dataset_<name>.json
    tests/conftest.py            writes small ann-benchmarks-format HDF5 files with exact ground truth
    tests/test_*.py              unit tests
    experiments/                 (planned) benchmark drivers
    run_all.sh, README.md, requirements.txt, hypothesis.md, IMPLEMENTATION_LOG.md, .gitignore, .gitattributes
```

Data flow (planned): `config` → `data_loader.load_dataset` → FAISS index builders (M3) → timing harness (M2)
→ `../results/results.csv` → plotting/table scripts → `../results/figures/`.

**Why data, results and ai_log are outside `code/`:** the zip needs `results/` and `ai_log/` as top-level items
(decision A4), and must not contain datasets. Since the scripts write straight to the top-level `results/`,
there is only one copy of every output. `code/.gitignore` also ignores `results/` and `ai_log/`, so stray copies inside the repo are never committed.

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

### Design revision between M1 and M2: approved decisions A1–A4 (2026-10-16)

**Status:** COMPLETED for the layout change (A4). A1–A3 are now **design specifications** (section 6) to be implemented in M2.
**Awaiting the author's review before M2 starts.**

**Trigger:** the author approved four decisions (CPU timing, recall@10 tie handling, index library, submission layout).

**Files modified:** `src/config.py` (RESULTS_DIR → `<repo>/../results`, overridable with `ANN_RESULTS_DIR`; shared
`_dir_from_env` helper), `tests/test_data_loader.py` (+1 test for the default results location), `.gitignore`
(ignores `results/` and `ai_log/` inside the repo), `README.md`, `IMPLEMENTATION_LOG.md`, `../ai_log/ai_statement.md` (paths).

**Files moved (not copied):** `code/results/` → `../results/` and `code/ai_log/` → `../ai_log/`. The previously committed
copies show as deletions in git, so the repo no longer contains them, as intended (they remain in commit `a17ac94`).
The empty `.gitkeep` placeholders were deleted because they are not needed outside git.

**Probe run:** the CPU-set mapping (section 2). Read-only, and not part of the code base yet.

**Commands executed**
```
python cpuset_probe.py                 # scratchpad; logical->core->EfficiencyClass mapping, affinity set/readback
mv code/results ./results; mv code/ai_log ./ai_log
bash run_all.sh                        # from code/: 19 passed; SIFT PASSED, GloVe PASSED; files in ../results/metadata/
```

**Tests (measured): 19 passed, 0 failed.** `run_all.sh` verifiably wrote `environment.json` and
`dataset_{sift,glove}.json` to `Assignment 3/results/metadata/`.

**What this establishes:** the output location follows the submission layout, and P-core pinning is technically possible on this machine.
**What it does not establish:** that pinning reduces timing variance. That will be measured in M2, not assumed.

### Design revision B: build-thread policy (A5) and hypothesis provenance (A6) (2026-10-16)

**Status:** COMPLETED as a plan, with a threading probe. **Awaiting the author's review. M2 not started.**

**Trigger:** the author approved multi-threaded builds on the verified P-core logical processors with single-threaded query
timing, plus a hypothesis guard and a result-provenance workflow. The author asked for the FAISS threading controls to be
*verified*, not assumed.

**Files created:** `experiments/probes/cpuset_probe.py`, `experiments/probes/thread_probe.py`, `experiments/probes/blas_probe.py`
(diagnostics, so the measurements below can be reproduced). **Files modified:** `IMPLEMENTATION_LOG.md`, `README.md`,
`../ai_log/ai_statement.md`. `experiments/.gitkeep` was removed (the folder is no longer empty).

**Threading probe: measured 2026-10-16.** Synthetic data: 60,000 × 64 float32 vectors, 2,000 queries. Process affinity was
restricted to logical processors 0–11 (the P-cores). These are not benchmark results.

*Native libraries bundled with faiss-cpu 1.15.1* (`faiss_cpu.libs/`):
- `vcomp140.dll`: the MSVC OpenMP runtime. `faiss.omp_set_num_threads` controls it.
- `libopenblas.dll`: **OpenBLAS 0.3.30**. Config string: `DYNAMIC_ARCH NO_AFFINITY Haswell MAX_THREADS=64`.
  Parallel mode 1 = its own thread pool, separate from OpenMP. **Default thread count: 20.**
  It exports `openblas_set_num_threads` / `openblas_get_num_threads`, which are callable through ctypes.
  `OPENBLAS_NUM_THREADS=1` set before import gives 1 thread (verified).
- NumPy loads a **separate** OpenBLAS (`libscipy_openblas64_…dll`) with its own pool. It is used only by our NumPy code, never inside FAISS.

*Thread affinity:* after the parallel work, the process had 56 threads. **Every thread's group-affinity mask was `0xfff`**,
i.e. confined to logical processors 0–11 (read with `GetThreadGroupAffinity`). The OpenMP and OpenBLAS worker threads therefore
inherit the process affinity, even though this OpenBLAS is built without its own affinity handling.

*Scaling, omp = OpenMP threads, blas = FAISS's OpenBLAS threads, cpu/wall = process CPU time ÷ wall time:*

| omp | blas | HNSW add (M=16) | IVF-PQ train+add (nlist 256, m 8) | Flat batch search, nq 2000 | Flat single-query ×200 |
|---|---|---|---|---|---|
| 1 | default (20) | 2.25 s | 5.10 s | 0.14 s | n/a |
| 12 | default (20) | 0.44 s (5.1×) | 5.42 s (0.9×) | 0.11 s | n/a |
| 1 | 1 | n/a | 5.50 s, cpu/wall 1.0 | 0.21 s, 0.9 | 0.09 s, 1.1 |
| 12 | 1 | n/a | 6.10 s, 11.0 | 0.20 s, 10.5 | 0.07 s, **8.7** |
| 1 | 12 | n/a | 4.96 s, 1.2 | 0.14 s, **1.9** | 0.09 s, 0.9 |
| 12 | 12 | n/a | 5.29 s, 11.1 | 0.11 s, 11.5 | 0.06 s, 11.8 |

*What the probe establishes:*
1. **No single setting controls all threading.** OpenMP and FAISS's OpenBLAS must both be set, and both read back.
2. **Single-query search is only single-threaded with omp = 1.** With omp = 12 it used several cores (cpu/wall 8.7).
3. **Batched exact search goes through BLAS.** With omp = 1 and blas = 12 its cpu/wall was 1.9, so for a single-threaded batch timing blas must also be 1.
4. HNSW construction scales with OpenMP: 5.1× with 12 threads on this small input.
5. With omp > 1, cpu/wall is about 11 even when wall time does not improve. OpenMP worker threads apparently spin while waiting
   for work, so **cpu/wall reliably detects "single-threaded" (about 1.0) but overstates useful parallelism.**
6. On this small input, IVF-PQ train+add did **not** speed up with threads (0.9–1.1×). **This does not establish** how it behaves at
   1M vectors, where assignment and encoding are larger parallel loops. That will be measured in M3 (see A5).

*Is the approved build configuration reliably supported?* **Yes for control and confinement.** Both thread counts can be set
and read back, and all threads stay on the 12 P-core logical processors. **Not yet known for speed-up:** some IVF build stages may not
benefit. That is a measurement to report, not a reason to change the policy. **No significant change to the approved policy is proposed.**

**Commands executed**
```
ls .venv/Lib/site-packages/faiss_cpu.libs                 # vcomp140.dll, libopenblas.dll, msvcp140.dll
.venv/Scripts/python experiments/probes/thread_probe.py   # DLLs, affinity masks, HNSW/IVF/Flat scaling
.venv/Scripts/python experiments/probes/blas_probe.py     # OpenBLAS config/controls, omp x blas matrix, cpu/wall
OPENBLAS_NUM_THREADS=1 python -c "...openblas_get_num_threads()"   # -> 1
```

### Milestone 2: Part 1 exact baseline (2026-10-17)

**Status:** COMPLETED. All Part 1 requirements have code, tests and measured results. **Awaiting the author's review before Milestone 3.**

**Commits:** design `d526e1e` → M2 code `45bed0f` → `run_all.sh` stage + pinning diagnostic `e098ca9`
(= the `git_commit` of every reported row) → this journal update (next commit). Results are hashed below (A6).

**Files created:** `src/recall.py`, `src/threads.py`, `src/timing.py`, `src/provenance.py`, `src/results_io.py`,
`experiments/__init__.py`, `experiments/part1_exact.py`, `experiments/probes/__init__.py`, `experiments/probes/pinning_probe.py`,
`tests/test_recall.py`, `tests/test_threads_timing.py`, `tests/test_provenance.py`, `../ai_log/milestone2/prompts.md`.
**Modified:** `src/data_loader.py` (`max_base` subset, `dataset_id`, `is_subset`; stored GT dropped for subsets),
`src/environment_info.py` (records topology and FAISS OpenBLAS config), `tests/test_data_loader.py` (+1 subset test), `run_all.sh`
(Part 1 stage, full and `--quick`), `README.md`, `IMPLEMENTATION_LOG.md`, `../ai_log/ai_statement.md`.
**Outputs (outside git):** `results/results.csv` (13 rows), `results/metadata/part1_{sift,glove}_reference.json`,
`results/metadata/pinning_probe.json`, logs `part1_full_run.log`, `pinning_probe.log`, `part1_run1_dirty_tree.log` (superseded run, kept for the record),
`results/quick/` (quick-mode outputs), and caches `data/cache/<dataset>_exact_top100.npz` (16 MB each, outside the zip, reused by M3).

**What was implemented**
1. `recall.py`: decision A2, exactly as specified.
   - `true_sq_distances` re-scores any id list in float64 (−1 → +inf, gathered in query chunks).
   - `reference_from_candidates` sorts candidates by float64 distance, then by id.
   - `recall_at_k` scores first occurrences only, ignores invalid ids, uses τ + 1e-6·max(τ, 1e-12), and always divides by k.
   - `boundary_tie_stats` reports tie counts, a tolerance-sensitivity table and gap statistics.
2. `threads.py`: decisions A1 and A5.
   - CPU-set topology comes from Windows (with a fallback elsewhere).
   - `build_phase()` uses all fastest-class CPUs, with OpenMP = OpenBLAS = their count.
   - `search_phase()` uses 1 pinned CPU with OpenMP = OpenBLAS = 1.
   - Calibration phases cover unpinned single-thread and multi-thread search.
   - Every setting is read back and raises an error on mismatch. `CpuWallTimer` measures CPU vs wall time.
3. `timing.py`: warm-up + R timed passes, median/min/max/all QPS, cpu/wall `thread_check` (limit 1.3), and the benchmark lock with stale-PID recovery.
4. `provenance.py`: `git_info`, `run_metadata` (run id, UTC time, HEAD, dirty, reportable) and `require_committed_hypothesis` (A6, implemented and
   tested now, used from M4). It also refuses cleanly when the file is not UTF-8.
5. `results_io.py`: the A6 column schema plus `search_mode` and `notes`. Rerunning replaces only rows with the same
   (experiment_id, dataset_id, n_base); unknown columns are rejected.
6. `experiments/part1_exact.py`:
   - builds `IndexFlatL2` under `build_phase()`;
   - computes an untimed exact top-100 for all 10,000 queries and re-scores it in float64 to get the reference τ;
   - checks FAISS and the stored ground truth against τ;
   - measures ties per split;
   - times five configurations (below), plus `--calibrate`;
   - `--quick` uses the first 100,000 base vectors and writes to `results/quick/`.

**Key decisions made during M2 (and why)**
- **FAISS 1.15.1's batch code path, a finding.** `faiss.cvar.distance_compute_blas_threshold` defaults to **128,000** here (measured).
  For any batch of fewer than 128,000 queries, FAISS therefore uses the per-query SIMD scan, parallelised over queries with OpenMP, not BLAS.
  A "batch" timing with default settings on one thread is the same as single queries (measured 0.99–1.01×).
  The driver therefore times both, with clear labels:
  - **primary batch** = BLAS path (threshold set to 20, the classic FAISS default), 1 thread;
  - **explanation row** = FAISS's default path, 1 thread;
  - **secondary** rows = both paths on the 12 P-core threads.

  This adds rows. It does not change the approved design (single-threaded primary comparison).
- **Query counts.** Single-query and default-path rows time the first **500** eval queries (≈13 s per pass at ≈38 QPS).
  BLAS-path rows time all **5,000** eval queries. Primary rows use R = 5 with 1 warm-up, secondary rows R = 3. Recall is
  computed for exactly the timed queries. All reported timings use eval queries only. The tune split was used for nothing.
- **Reference = FAISS exact top-100 re-scored in float64.** The τ for every later milestone comes from this cached reference. Batched search in
  FAISS computes ‖x‖² + ‖q‖² − 2q·x in float32, which can reorder near-equal neighbours; float64 re-scoring removes that.
- **Index size** = `len(faiss.serialize_index(index))`. For Flat: SIFT 512,000,045 B = 512.0 B/vector (128 × 4 B + 45 B header);
  GloVe 473,405,645 B = 400.0 B/vector. "Build time" for Flat is only copying vectors (0.15 s SIFT, 0.14 s GloVe) and is not meaningful.

**Commands executed**
```
.venv/Scripts/python -m pytest -q                                   # 54 -> 55 passed (see section 10)
python - (20-query estimate)                                        # sizing: 41 qps single, 38 qps "batch" -> led to the finding above
python - (BLAS-threshold diagnosis, 200 queries)                    # threshold 128000; 1T BLAS 340 qps vs default 38.8
.venv/Scripts/python -m experiments.part1_exact --quick --calibrate --n-single 200 --repeats 3 --secondary-repeats 2 --calibration-queries 100   # 33 s
git commit 45bed0f (code)
.venv/Scripts/python -m src.environment_info
.venv/Scripts/python -m experiments.part1_exact --calibrate         # run 1: rows marked git_dirty -> superseded (bug 3)
git commit e098ca9 (run_all.sh, pinning probe)
.venv/Scripts/python -m experiments.probes.pinning_probe            # about 4 min
.venv/Scripts/python -m experiments.part1_exact --calibrate         # run 2 (reported), about 12 min
sha256sum results/...                                               # provenance hashes below
```

**Measured results (run 2, commit `e098ca9`, 2026-10-17; all rows `git_dirty=False`, `reportable=True`; single-thread rows `thread_check=PASS`, cpu/wall 0.97–0.99)**

*Ground-truth agreement (all 10,000 queries; τ = float64 10th-nearest distance from our exact search):*

| | SIFT | GloVe |
|---|---|---|
| FAISS exact top-10, recall@10 vs τ | 1.000 (0 queries < 1) | 1.000 (0 queries < 1) |
| Stored ground-truth top-10, recall@10 vs τ | **1.000** (0 queries < 1) | **1.000** (0 queries < 1) |
| Queries where the stored 10th distance ≠ our τ (beyond tolerance) | 0 | 0 |
| Queries whose stored top-10 **id set** differs from ours | **65** | **1** |
| …of which every difference is a boundary tie (recall 1.0) | 65 | 1 |
| FAISS float32 top-10 id set ≠ float64-re-scored top-10 | 0 | 1 (within tolerance) |
| Reference computation time (10,000 × top-100, 12 threads, untimed for QPS) | 15.3 s | 19.7 s |

→ **Our exact search agrees with the stored ground truth on every query.** Comparing id sets naively would have reported 65 SIFT and 1 GloVe
"disagreements". Every one of them is a tie at the 10th neighbour, which is exactly what A2 is designed to handle.

*Boundary ties (rank 11 tied with rank 10), measured, not hardcoded:*

| | SIFT all / tune / eval | GloVe all / tune / eval |
|---|---|---|
| Tied at rel_tol = 1e-6 | **137** / 74 / 63 | **2** / 1 / 1 |
| Sensitivity (tolerance 0 / 1e-7 / 1e-6 / 1e-5 / 1e-4) | 137 / 137 / 137 / 137 / 226 | 0 / 1 / 2 / 18 / 195 |
| Smallest positive relative gap d₁₁−d₁₀ | 1.34e-5 | 2.1e-10 |
| Smallest gap above tolerance | 1.34e-5 | 2.53e-6 |

Interpretation:
- **SIFT:** squared distances are integers, so ties are exact. The count is stable from tolerance 0 to 1e-5, and the nearest non-tie is 13× above the tolerance.
- **GloVe:** distances are continuous, so the tie count *depends on the tolerance*. The 2.1e-10 gap is far below what float32-normalised data can
  resolve. M1 saw it as an exact tie in the stored float32 distances, which fits. The smallest gap above the tolerance (2.5e-6) is only 2.5× the
  tolerance, so the boundary is not cleanly separated on GloVe.
- Practical effect: at most 2 of 10,000 GloVe queries are affected, which cannot move any reported recall noticeably.

*Exact search timing (eval queries, median of R passes):*

| Row (role) | Threads | Queries | SIFT QPS (min–max) | GloVe QPS (min–max) |
|---|---|---|---|---|
| Single query per call (**primary**) | 1, pinned cpu 10 | 500 | **38.05** (37.7–38.4) → 26.3 ms/q | **39.98** (39.7–40.7) → 25.0 ms/q |
| Batch, BLAS path (**primary batch**) | 1, pinned | 5,000 | **325.97** (323.5–332.7) = **8.57×** | **334.04** (332.0–337.9) = **8.36×** |
| Batch, FAISS default path (explanation) | 1, pinned | 500 | 37.69 = 0.99× single | 40.30 = 1.01× single |
| Batch, BLAS path (secondary) | 12 (P-cores) | 5,000 | 669.78 = 2.05× BLAS-1T | 658.59 = 1.97× BLAS-1T |
| Batch, FAISS default path (secondary) | 12 (P-cores) | 500 | 186.36 = 4.95× default-1T | 235.33 = 5.84× default-1T |

*Why batched exact search is about 8.5× faster on one thread (mechanism, supported by the measurements):*
1. One single query must stream the **whole** base through the core: 0.512 GB (SIFT) or 0.473 GB (GloVe). It does only about 2 floating-point
   operations per 4-byte value. The measured rates imply **19.5 GB/s** (SIFT) and **18.9 GB/s** (GloVe) of data read and only about 9.6 GFLOP/s.
   The two datasets have different n and d but read data at almost the same rate, which is consistent with a **memory-bandwidth-bound** scan.
   *Not verified:* we did not run a separate memory-bandwidth benchmark.
2. The BLAS path turns the batch into a matrix product (−2·Q·Xᵀ). Each block of base vectors is loaded into cache once and reused against many
   queries, so the work becomes **compute-bound**: 83.4 (SIFT) and 79.1 (GloVe) GFLOP/s, about 8.7× more arithmetic per second.
   The measured speed-up equals that factor.
3. Simply sending many queries in one call gives nothing (default-path row: 0.99–1.01×). The Python call overhead is negligible compared with a
   26 ms scan. **The gain comes from data reuse (the GEMM algorithm), not from batching the calls.**
4. With 12 threads, the BLAS path gains only about 2× more (cpu/wall about 11.7, so most of the extra CPU time is spent waiting). The cause is
   *not determined*: possible factors include shared memory bandwidth, 6 physical cores (12 threads are SMT pairs), and FAISS's
   top-k selection step. This row is secondary and is not used for any QPS table.

*CPU placement (calibration + diagnostic; SIFT, single-query, 1 thread):*

| | Calibration (200 q × 5) | Pinning diagnostic (100 q × 2, two rounds in opposite order; mean of round medians) |
|---|---|---|
| Pinned P-core (cpu 10) | 38.3 QPS (spread 0.9%) | cpu 10: 37.9; all 12 P-core CPUs: 37.6–38.5 |
| Pinned E-core (cpu 12) | 28.6 QPS (spread 1.3%) | cpu 12: 28.7, cpu 13: 28.8 |
| Unpinned, 1 thread | 32.4 QPS (spread 6.9%) | 34.9 |

What this shows:
- **Pinning to a P-core gives the highest and most stable QPS.**
- An E-core is about 25% slower.
- Unpinned runs land in between and vary more, which fits Windows sometimes moving the thread to an E-core. This is an inference; scheduling
  was not traced.
- Run 1's calibration showed the opposite (unpinned 37.7 > pinned 35.8). That **did not repeat** in run 2 or in the diagnostic, and is treated
  as run-to-run variation (see the next point).
- **No change to the A1 policy is needed.**

*Run-to-run variation (not just within-run spread):* the same configuration measured in run 1 → run 2 (different commits, same code) gave
SIFT single 36.26 → 38.05 (+4.9%) and GloVe single 40.66 → 39.98 (−1.7%). Within a run, the spread is about 1–3%.
**Between-run variation of about 5% is larger than within-run spread.** QPS differences under 5% between configurations measured in
different runs should not be interpreted (relevant to Q5).

**Provenance (A6), SHA-256 of outputs at this commit:**
```
5521e3871e6bb768b881148f71f40d0dd07132a4785bb979eaec7e1bdfe4bb6a  results/results.csv
8f2a47c1fd1a47a50825099b69fdcaa7707e257a10b1d72a00be2acbf1351d03  results/metadata/part1_sift_reference.json
97a351b10381805eb4ae263934f9d4cc60d8cb52781f9ce40feb2419f48563ed  results/metadata/part1_glove_reference.json
0c2b7bd4d957f4015e2a4db17fa87fe0c2912fffce79f242ca2ba639d94ec05b  results/metadata/pinning_probe.json
7ff7bfca5ec02cf6da01ed0d7cdf3e4c8f320edc2a4b3790353ec6765029655a  results/metadata/environment.json
```

**What M2 establishes:**
- The recall function is correct on 16 targeted unit tests and gives the expected results on real data.
- FAISS exact search equals the stored ground truth under tie-aware scoring.
- Single-query vs batched exact QPS on this machine, and the mechanism for the difference (data reuse).
- P-core pinning is the most stable timing setup.
- The thread controls hold during real runs (cpu/wall ≈ 1.0 on every single-thread row).

**What it does not establish:**
- The memory-bandwidth explanation is inferred from rates, not measured with a bandwidth benchmark.
- Why 12-thread BLAS scales only 2×.
- Behaviour on other machines, or under the `--quick` subset at scale (quick outputs exist but are not report numbers).

**Remaining / next:** Milestone 3 (Part 2 ANN benchmarks), after the author's review.

### Milestone 3: Part 2 ANN benchmarks (2026-10-17)

**Status:** COMPLETED. All Part 2 requirements have code, tests and measured results. **Awaiting the author's review before Milestone 4.**

**Commits:**
- `55016fe` hypotheses (author) → `e341426` M3 code → `78e0e5f` fix for a syntax error in the report script + compile test
  (= `git_commit` of all 367 `p2_ann` rows) → `77f89d2` table-column fix → `398f56f` build-scaling script (= `git_commit` of the 6 `p2_build_scaling` rows)
  → this journal update.
- The report script has no provenance rows; the figures and tables were regenerated after that commit and are hashed below.

**Files created:** `src/indexes.py`, `src/pareto.py`, `experiments/part2_ann.py`, `experiments/part2_report.py`,
`experiments/build_scaling.py`, `tests/test_part2.py`, `tests/test_scripts_compile.py`, `../ai_log/milestone3/{prompts,response}.md`.
**Modified:** `src/results_io.py` (configurable replacement key), `src/provenance.py` + `tests/test_provenance.py` (numbered-heading
fix, commit `468fb36`), `run_all.sh` (Part 2 stages), `requirements.txt` (+ matplotlib 3.11.2), `README.md`, `IMPLEMENTATION_LOG.md`, `../ai_log/ai_statement.md`.
**Outputs (outside git):**
- `results/results.csv` (now 386 rows)
- `results/figures/part2_{sift,glove}_recall_vs_qps.png`
- `results/tables/part2_{qps_at_recall,build_cost_at_095,max_recall}.{csv,md}`
- `results/metadata/part2_{sift,glove}_builds.json` (build-time breakdown)
- logs `part2_sift_HNSW.log`, `part2_sift_IVF.log`, `part2_glove.log`, `build_scaling.log`
- quick-mode outputs in `results/quick/`

**What was implemented**
1. `src/indexes.py`: FAISS builders, all `METRIC_L2`.
   - HNSW: `IndexHNSWFlat(d, M)` with efConstruction 200.
   - IVF-Flat and IVF-PQ: built on a pre-trained `IndexFlatL2` coarse quantizer. IVF-PQ uses 8 bits per code and residual encoding (FAISS default).
   - k-means is seeded (`cp.seed = 12345`) and computed **once per nlist per dataset**, then shared by IVF-Flat and IVF-PQ.
     Every IVF build's reported time includes the full k-means time. A test checks that shared and standalone centroids give identical search results.
   - Size = `len(faiss.serialize_index())`.
2. `src/pareto.py`: Pareto front over (recall, QPS), and "fastest configuration whose recall ≥ target". A test checks that the target pick always lies on the front.
3. `experiments/part2_ann.py`: the protocol below. `experiments/part2_report.py`: figures and tables built only from `results.csv`.
4. `experiments/build_scaling.py`: the A5 check comparing P-core-threaded and single-threaded builds.

**Protocol actually used (tune → select → eval)**

| Step | Queries | What is measured | Setting |
|---|---|---|---|
| Tune sweep | tune 0–4999 | recall@10 on all 5,000 (batch search, untimed); QPS = single-query, 1 thread, pinned cpu 10, on tune[0:1000], R = 3 + warm-up | every build × search value |
| Select | (tune rows only) | Pareto front per index type; for 0.90 and 0.95, the fastest configuration with tune recall ≥ target | |
| Eval | eval 5000–9999 | recall@10 on all 5,000; QPS on eval[0:1000], R = 5 + warm-up | only the selected configurations |

Sweep lists: efSearch {10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024}; nprobe {1, 2, 4, …, 1024} (never more than nlist).

Early stopping:
- tune recall ≥ 0.995;
- tune QPS below exact-search QPS (from Part 1);
- recall plateau (two consecutive steps each gaining less than 0.002).

Build grid:

| Index | SIFT | GloVe |
|---|---|---|
| HNSW | M ∈ {8, 16, 32}, efC 200 | same |
| IVF-Flat | nlist ∈ {1024, 4096, 16384} | same |
| IVF-PQ | nlist ∈ {1024, 4096} × m ∈ {16, 32, 64} | nlist ∈ {1024, 4096} × m ∈ {10, 25, 50} (m must divide d = 100) |

Validation inside every measurement:
- the single-query ids on the timed subset are compared with the batch ids;
- **all 367 rows match**, so recall from batch search is exactly the recall of the timed single-query search;
- all 367 rows pass `thread_check` (cpu/wall 0.71–1.14);
- all rows are `reportable=True` from commit `78e0e5f`.

**Commands executed**
```
pip install matplotlib (first attempt stalled on the network for over 5 min; stopped and retried with --timeout 30)
.venv/Scripts/python -m pytest -q                                  # 68 -> 89 -> 90 passed
.venv/Scripts/python -m experiments.part2_ann --quick ... ; part2_report --quick   # 45 s end to end
git commit e341426; git commit 78e0e5f (syntax fix)
.venv/Scripts/python -m experiments.part2_ann --dataset sift --index HNSW          # pilot, 4.7 min
.venv/Scripts/python -m experiments.part2_ann --dataset sift --index IVF-Flat --index IVF-PQ   # 24.0 min
.venv/Scripts/python -m experiments.part2_ann --dataset glove                      # 52.9 min
.venv/Scripts/python -m experiments.part2_report
.venv/Scripts/python -m experiments.build_scaling --quick ; git commit 398f56f ; build_scaling (full, about 9 min)
sha256sum results/...
```
The author approved the 60–90 minute full run after the pilot (AskUserQuestion: "Run everything now"). Total measured run time ≈ 82 min.

**Measured results (eval queries; configuration chosen on tune queries)**

*QPS at recall 0.90 / 0.95 (`results/tables/part2_qps_at_recall.md`):*

| Dataset | Index | @0.90: config → eval recall, QPS | @0.95: config → eval recall, QPS |
|---|---|---|---|
| SIFT | HNSW | M=32, ef=24 → 0.9155, **11,626** | M=32, ef=48 → 0.9711, **6,783** |
| SIFT | IVF-Flat | nlist=4096, nprobe=32 → 0.9333, 3,055 | nlist=16384, nprobe=128 → 0.9733, 1,709 |
| SIFT | IVF-PQ | not reached (max tune recall 0.884) | not reached |
| GloVe | HNSW | M=32, ef=192 → 0.9179, **1,523** | M=32, ef=384 → 0.9546, **762.5** |
| GloVe | IVF-Flat | nlist=16384, nprobe=512 → 0.9241, 752.4 | nlist=16384, nprobe=1024 → 0.9555, 412.1 |
| GloVe | IVF-PQ | not reached (max tune recall 0.869) | not reached |
| SIFT / GloVe | Exact (Part 1) | 38.05 / 39.98 QPS at recall 1.0 | |

Every selected configuration also meets its target on eval.

*Build cost of the recall-0.95 configuration (`part2_build_cost_at_095.md`):*

| Dataset | Index | Build time (12 P-core threads) | Index size | Bytes/vector |
|---|---|---|---|---|
| SIFT | HNSW M=32 | 82.6 s | 784.1 MB | 784.1 |
| SIFT | IVF-Flat nlist=16384 | 321.8 s (k-means 291.9 s) | 528.5 MB | 528.5 |
| GloVe | HNSW M=32 | 135.2 s | 795.5 MB | 672.1 |
| GloVe | IVF-Flat nlist=16384 | 367.6 s (k-means 334.0 s) | 489.6 MB | 413.6 |
| both | IVF-PQ | 0.95 not reached: 18–74 B/vector, builds 17–137 s | | |

Reference: the raw vectors are 512 B (SIFT) and 400 B (GloVe). Flat = 512.0 / 400.0 B/vector.

*Highest recall reached (`part2_max_recall.md`), and why:*

| Index | SIFT | GloVe | Why it stopped |
|---|---|---|---|
| HNSW | 0.995–0.996 for all M | M=8: 0.866, M=16: 0.951, M=32: 0.984 | SIFT: stop rule (≥ 0.995). GloVe: **end of the efSearch list (1024)**, still rising slowly, so these are sweep limits, not proven ceilings |
| IVF-Flat | 0.996–0.999 | nlist 1024: 0.997, 4096: 0.987, 16384: 0.956 | GloVe nlist 4096/16384: **end of the nprobe list (1024)** = 25% / 6% of lists, so also sweep limits |
| IVF-PQ | m=16: 0.56–0.59, m=32: 0.73–0.74, m=64: 0.88 | m=10: 0.31–0.32, m=25: 0.62–0.63, m=50: 0.87 | **genuine plateau**: all 6 SIFT sweeps and 2 GloVe sweeps stopped on the plateau rule; GloVe m=50 at nprobe 256/512/1024 gives 0.860/0.868/0.869 |

*Why IVF-PQ cannot reach high recall (mechanism; supported by the data, with one inference marked):*
- IVF-PQ ranks candidates by **approximate** distances computed from 8-bit codes for each sub-vector.
- Raising nprobe only adds candidates to the pool. It cannot fix the ordering error of the PQ distances, so recall stops rising once the
  true neighbours are already in the scanned lists (the plateau above).
- The plateau level depends almost only on m, not on nlist. SIFT: nlist 1024 vs 4096 differ by ≤ 0.02, while m = 16 → 32 → 64 raises it 0.56 → 0.73 → 0.88.
  More sub-vectors means finer quantization of each part of the vector, so the distance errors are smaller.
- *Inference, to be tested in Part 3:* the true neighbours are in the candidate pool but mis-ranked. Part 3's re-ranking of the top 100
  candidates tests this directly (hypothesis 3.2, committed in `55016fe`).

*Answer to Q1 (preliminary, for the report):*
- **HNSW is fastest at recall 0.95 on both datasets.** It beats IVF-Flat by **3.97×** on SIFT (6,783 vs 1,709 QPS) and by **1.85×** on GloVe (762.5 vs 412.1).
- The ranking does not change between datasets, but the **margin and absolute speed do**. At recall 0.95, HNSW is 8.9× slower on GloVe
  (6,783 → 762.5 QPS) and IVF-Flat 4.1× slower (1,709 → 412.1), which is why the HNSW lead shrinks from 3.97× to 1.85×. GloVe needed ef = 384 against 48 on SIFT, and nprobe = 1024 against 128.
- Speed-up over exact search at 0.95: SIFT HNSW 178×, GloVe HNSW 19×.
- *Explanation not yet established:* why GloVe is harder (for example, lower relative contrast between neighbours and other points).
  Not measured here. Part 4 (dimensionality) and an optional relative-contrast measurement could support it.

*Build scaling (A5; SIFT; `p2_build_scaling`, commit `398f56f`):*

| Configuration | 12 P-core threads | 1 thread (pinned) | Speed-up | Breakdown |
|---|---|---|---|---|
| HNSW M=16 | 69.8 s | 380.3 s | **5.45×** | graph insertion parallelises well (≈ 6 physical cores) |
| IVF-Flat nlist=1024 | 7.9 s | 13.9 s | 1.76× | k-means 5.8 vs 9.9 s, add 2.1 vs 4.0 s |
| IVF-PQ m=16, nlist=1024 | 21.8 s | 38.2 s | 1.75× | **PQ training 12.1 vs 11.7 s (no speed-up)**, encode 4.0 vs 16.5 s (4.1×) |

What this means:
- The reported build times favour HNSW relative to a single-threaded comparison. At 1 thread, the HNSW M=16 build would cost about 380 s.
- The ranking of IVF build costs barely depends on threads.
- The 12-thread times repeat the Part 2 build times closely (IVF-PQ 21.8 = 21.8 s, IVF-Flat 7.9 vs 6.8 s, HNSW 69.8 vs 70.1 s).

*Tune vs eval consistency (configurations measured on both splits):*
- **Recall:** on SIFT, eval recall was higher than tune recall for **all 47 selected configurations** (mean +0.0156, range +0.0003 to +0.0483).
  On GloVe there is no systematic difference (mean +0.0024, range −0.0051 to +0.0073). So the last 5,000 SIFT queries are measurably easier
  than the first 5,000. The reason is not known. Effect: tune-based selection is slightly **conservative** on SIFT. For example, HNSW M=16, ef=48
  scored tune 0.9454 (so it was not chosen for 0.95) but eval 0.9533.
- **QPS:** median eval/tune ratio 0.998, range 0.66–1.18. Individual configurations can differ by up to about 34% between the two measurements
  (different queries, R = 3 vs 5, run-to-run variation). Reported QPS comes only from the eval rows.
- The plotted curves are eval measurements of the **tune**-selected front, so they have small non-monotonic steps (GloVe HNSW near 0.81,
  SIFT IVF-Flat near 0.93). This is expected and is not corrected, to keep the tune/eval separation.

**Provenance (A6), SHA-256 of outputs at this commit:**
```
5d92e372ae7871c426e7f686c9505444f8464caaedb0e3855e0e7aa49830482d  results/results.csv
fbdfce62e1b4b7c30fd5d92768653d30f91ed61a998380f3ddbc231ac0dadbe7  results/figures/part2_sift_recall_vs_qps.png
b1ef8bcb0da92d827caf42cb8bc65f1f322532e2113a0ab898d7fc0c84a76eca  results/figures/part2_glove_recall_vs_qps.png
d6a0bf5ce07eea90c1ae0c885bd859c91f900877c9f39f4ccd6d0353bad9da05  results/tables/part2_qps_at_recall.csv
6be7e751a85e11ed8bbcf5ead5b6e6fe26128876a1c0ae10f5bc3effcab306ef  results/tables/part2_build_cost_at_095.csv
ea68705ae9cf9aa8723fc7f88db5b7ed5d446c42d0b0d939f730323f208968a8  results/tables/part2_max_recall.csv
8ea7fd8f9df0278b3a2fd0eb8df84b1bf38f07d6c8ed02b1e72d0f68b4e83c91  results/metadata/part2_sift_builds.json
85f4b62e190978cc562e5fce184d28c2ce4555972f0e2aaa1758411f60addf4e  results/metadata/part2_glove_builds.json
```

**What M3 establishes:**
- On this machine, under this protocol, HNSW gives the highest single-thread QPS at recall 0.90 and 0.95 on both datasets.
- IVF-Flat reaches high recall but needs far larger nprobe on GloVe.
- Plain IVF-PQ cannot reach 0.90 at any tested code size (up to 64 B on SIFT, 50 B on GloVe), and its ceiling is set by m.
- PQ shrinks the index 6.9–22.5× compared with IVF-Flat on the same dataset (18.4–74.3 vs 408–529 B/vector).
- Build cost: HNSW 47–135 s (graph links add 80.6 B/vector at M=8 up to 272 B/vector at M=32); IVF dominated by k-means, which grows steeply with nlist (SIFT: 5 s → 74 s → 292 s).

**What it does not establish:**
- True recall ceilings for GloVe HNSW M ≤ 16 and IVF-Flat nlist ≥ 4096 (sweep-limited).
- Effects of efConstruction (fixed at 200), of nbits ≠ 8, or of OPQ.
- Multi-threaded or batched QPS.
- Why the SIFT query halves differ in difficulty.
- Why GloVe is harder.

**Remaining / next:** Milestone 4 (Part 3 compression experiment, guarded by hypothesis commit `55016fe`), after the author's review.

### Milestone 4: Part 3 compression experiment (2026-10-22)

**Status:** COMPLETED. **Awaiting the author's review before Milestone 5 (Part 4).**

**Provenance (A6):**
- Hypothesis commit **`55016fe`** (author-written, pushed 2026-10-17).
- → M4 code `b230f71` (= `git_commit` of all 158 `p3_compression` rows; every row also stores
  `hypothesis_commit = 55016fe` and the Part 3 section SHA-256 `f6912f6127119b01…`).
- → this journal commit.
- The guard ran first in every run:
  - It **refused** the first quick attempt (dirty tree, exit 3).
  - It allowed the `--dev` quick run (rows in `results/quick/`, not reportable).
  - It passed for the full run on the clean tree.

**Files created:** `experiments/part3_compression.py`, `experiments/part3_report.py`, `tests/test_part3.py`, `../ai_log/milestone4/{prompts,response}.md`.
**Modified:**
- `src/recall.py`: `pool_recall`; recall logic factored into `_correct_mask`, so the two cannot diverge.
- `src/indexes.py`: `add_exact_refine`.
- `src/results_io.py`: columns `pool_recall_10at100`, `candidate_qps_median`, `rerank_ms_per_query`.
- `run_all.sh`: Part 3 stages, plus build scaling in the full run.
- `README.md`, `IMPLEMENTATION_LOG.md`, `../ai_log/ai_statement.md`.

**Outputs (outside git):**
- `results/results.csv` (now 544 rows)
- `results/figures/part3_compression.png`
- `results/tables/part3_by_m.{csv,md}`, `results/tables/part3_rerank_cost.{csv,md}`
- `results/metadata/part3_sift_builds.json`, `results/metadata/part3_sift.log`
- `results/quick/` (quick run)

**Design (one property varied: the number of PQ sub-vectors m)**

| Fixed | Varied | Compared |
|---|---|---|
| SIFT-1M; nlist = 4096 (one seeded k-means, shared); nbits = 8; same tune/eval split and protocol as Part 2 | m ∈ {8, 16, 32, 64} (code = m bytes) | **plain** IVF-PQ (top 10 by PQ distance) vs **+ exact re-ranking** (`IndexRefineFlat`, k_factor = 10: PQ top 100, exact L2, best 10) |

How the comparison is kept fair:
- Both variants use **the same built IVF-PQ index**, verified in a pre-check and in `test_part3.py`, so at any nprobe the re-ranked
  answer is computed from exactly the candidate pool that plain search with k = 100 returns. This is what hypothesis 3.2 asked for
  ("using the same candidate pool").
- For every row we also measure **pool recall 10@100**: the tie-aware fraction of the true top 10 present anywhere in the 100 candidates.
  Eval rows of the re-ranked variant additionally time candidate generation alone (plain, k = 100). This gives
  **re-rank cost = 1/QPS(refine) − 1/QPS(candidates)**.
- Selection: per (variant, m), the tune Pareto front plus the fastest configuration reaching recall 0.50 / 0.80 / 0.90 / 0.95. Only those are evaluated on eval.

**Commands executed**
```
python - (IndexRefineFlat pre-check, 20K random vectors)        # shares base, k_factor 10 = top 100, equals manual re-rank
.venv/Scripts/python -m pytest -q                                # 90 -> 100 passed (see bug 2)
.venv/Scripts/python -m experiments.part3_compression --quick    # guard REFUSED (dirty tree), exit 3, as designed
.venv/Scripts/python -m experiments.part3_compression --quick --dev ...   # 2.0 min; rows in results/quick (not reportable)
.venv/Scripts/python -m experiments.part3_report --quick
git commit b230f71
AskUserQuestion -> "Run it now (Recommended)" (estimate 25-40 min)
.venv/Scripts/python -m experiments.part3_compression           # 21.9 min, exit 0
.venv/Scripts/python -m experiments.part3_report
sha256sum results/...
```

**Measured results (SIFT-1M, nlist 4096; eval queries; configurations chosen on tune)**

*Per m (`results/tables/part3_by_m.md`):*

| m | B/vector plain | B/vector + re-rank | Build (12 threads) | Best eval recall, plain | Best eval recall, re-ranked (= pool 10@100) | QPS @0.80 plain / re-ranked | QPS @0.90 re-ranked | QPS @0.95 re-ranked |
|---|---|---|---|---|---|---|---|---|
| 8 | 18.3 | 530.3 | 83.4 s | 0.4148 | 0.8879 | not reached / 5,568 (r = 0.859) | not reached | not reached |
| 16 | 26.3 | 538.3 | 89.0 s | 0.5967 | 0.9861 | not reached / 5,841 (0.853) | **4,103** (0.927, nprobe 32) | **2,663** (0.967, nprobe 64) |
| 32 | 42.3 | 554.3 | 103.0 s | 0.7462 | 0.9991 | not reached / 4,107 (0.856) | 2,707 (0.933) | 1,637 (0.977) |
| 64 | 74.3 | 586.3 | 128.0 s | 0.8861 | 0.9993 | **1,729** (0.849) / 2,650 (0.856) | 1,627 (0.933) | 931 (0.977) |

Build-time breakdown:
- Shared k-means: 67.2 s.
- PQ training: 6.5 / 12.4 / 24.0 / 46.5 s, linear in m.
- Adding vectors: 9.4–14.4 s.
- Copying the raw vectors for re-ranking: 0.14–0.16 s.

Bytes per vector:
- Plain = **m + 10.3 B** exactly for every m. The extra 10.3 B is an 8-byte id plus about 2.3 B of centroids and list overhead per vector.
- Re-ranked = **plain + 512.0 B**, the float32 raw vector kept in RAM.

*Cost of exact re-ranking (eval, same m and nprobe; `part3_rerank_cost.md`):*
- The exact re-ranking step itself costs **0.012–0.030 ms per query** at every m and nprobe.
  That is about 100 random-access reads of 512-byte vectors plus 100 exact distances, i.e. constant work independent of the scan size.
  It is 9–15% of latency at nprobe ≤ 8 and 1–6% at nprobe ≥ 64.
- Asking the PQ index for 100 instead of 10 candidates adds another 0.006–0.10 ms (larger result heap).
- At the same nprobe, the re-ranked variant therefore runs at **0.79–1.02× the plain QPS** (tune rows, nprobe 8 / 32 / 128).
- The subtraction is unreliable when the total latency is large. One value, m = 16 at nprobe 1024, came out as 0.107 ms at about 4 ms total latency.

*Check of the mechanism:* **re-ranked recall@10 equals pool recall 10@100 in 78/78 re-ranked rows** (maximum difference 0.000000).
Every re-ranked configuration also has recall ≥ plain recall at the same (m, nprobe).

**Hypotheses vs results** (hypotheses written by the author in `55016fe`; each verdict is the reading of the measured data, for the author to confirm)

| Hypothesis | Prediction | Measured | Reading |
|---|---|---|---|
| **3.1** more sub-vectors → better recall, more storage; gains may diminish | recall ↑ with m; storage ↑; diminishing returns | Plain best recall 0.415 → 0.597 → 0.746 → 0.886, rising at every m. Storage = m + 10.3 B, linear. Recall gained per extra byte: 0.0227 → 0.0093 → 0.0044 per byte (m 8→16→32→64), a 5× drop. With re-ranking, recall saturates: 0.888 → 0.986 → 0.999 → 0.999 | **Supported.** Also observed, not predicted: QPS at a fixed nprobe falls as m rises (plain at nprobe 32 on tune: 6,993 → 4,484 → 3,244 → 1,737), because computing PQ distances costs O(m) |
| **3.2** exact re-ranking of top 100 improves or preserves recall; cannot recover neighbours absent from the pool | re-ranked ≥ plain; bounded by the pool | Re-ranked ≥ plain in every pair. Re-ranked = pool recall exactly (78/78 rows). Largest gains: m = 8 0.415 → 0.888, m = 16 0.597 → 0.986. The m = 8 shortfall (0.888) equals its pool recall, i.e. the remaining misses are neighbours that are absent from the top 100 | **Supported**, including the "cannot recover absent neighbours" clause, which was measured directly |
| **3.3** re-ranking reduces QPS, but may give a better recall–speed trade-off | QPS ↓ at equal settings; better at similar recall | At equal (m, nprobe): QPS 0.79–1.02× plain. At similar recall it is much better: ≈ 0.85 → plain needs m = 64 (1,729 QPS), re-ranked m = 16 gives 5,841 QPS (**3.4×**). Plain cannot reach 0.90 at all. Re-ranking cost measured separately: 0.012–0.030 ms per query | **Supported** for speed. **Not in the hypothesis:** the gain is bought with memory, +512 B/vector (530–586 vs 18–74 B), which removes PQ's memory advantage when the raw vectors are kept in RAM |

*What this implies (inference, for report Q2/Q3):*
- With re-ranking, a **small code (m = 16) is the best choice**. Larger m only slows queries (m = 16 vs 64 at 0.95: 2,663 vs 931 QPS) for ≤ 0.013 extra recall.
- What limits accuracy moves from code size (plain) to **candidate-pool coverage**, which is controlled by nprobe and k_factor.
- PQ + re-ranking is **not** a memory saving if the full vectors stay in RAM: 538 B/vector vs IVF-Flat 528.5.
- Its benefit is speed. At 0.95 it gives 2,663 QPS, against Part 2's IVF-Flat 1,709 (nlist 16384) and HNSW 6,783 (784 B/vector).
  These are measured in different runs, so differences under about 5% would not be meaningful (M2).
- A memory saving needs the raw vectors on disk or SSD, or a smaller refine store (for example SQ8). **Not tested.**

**Anomaly (kept, not corrected):**
- Eval row plain m = 64, nprobe 4: **3,357 QPS** (5 repeats 3,354–3,386, cpu/wall 1.00).
- Tune measured **6,472 QPS** for the same configuration, and its own k = 100 candidate search measured 0.169 ms against its 0.298 ms.
- The repeats are internally consistent, so something slowed the whole roughly 2-second measurement. The cause is unknown.
- It affects one table cell: QPS@0.50 for plain m = 64. No hypothesis verdict depends on it.
- Following the protocol, it was neither re-measured nor removed. A re-check is possible on request.
- Separately, 7 of 158 rows (all tune) have a spread between repeats above 10% (maximum 34%). Medians are reported.

**Provenance, SHA-256 of outputs at this commit:**
```
89472bd8ee77b5033bdaeae0020b8f45d43c246df6f4692e233d5e2eb8769699  results/results.csv
de977dda397abdfc1a11f57e4bf52e754a542f7ee35d600bef3437249bde6c12  results/figures/part3_compression.png
184cbbbe28636bb5e394097efbdbd0202b0c2cf3d99e0fd5aa404508e72aec77  results/tables/part3_by_m.csv
3235cfe125638cfa5a431bb566057f52ed7fa9718d5e62a305e7dde4bf649be3  results/tables/part3_rerank_cost.csv
4646417ea287164b8e9d965a3c75b032e4fe65158724b64f1f1f3b2502033312  results/metadata/part3_sift_builds.json
```

**What M4 establishes:**
- On SIFT-1M, nlist 4096: plain IVF-PQ recall rises with m, with diminishing returns per byte.
- Exact re-ranking of the top 100 reaches exactly the candidate-pool recall, 0.89–0.999.
- The re-ranking step costs about 0.02 ms per query.
- At matched recall, re-ranking makes IVF-PQ several times faster, but costs +512 B/vector.

**What it does not establish:**
- Other nlist values or k_factors (only 10 was tested).
- Other datasets (Part 3 is SIFT only, as specified).
- Refine stores smaller than float32, or on disk.
- Multi-threaded behaviour.
- The cause of the one anomalous timing.

**Remaining / next:** Milestone 5 (Part 4: dimensionality on seeded synthetic data, guarded by `55016fe`), after the author's review.

### Milestone 5: Part 4 dimensionality on seeded synthetic data (2026-10-22)

**Status:** COMPLETED. **Awaiting the author's review before Milestone 6 (report and submission).**

**Provenance (A6):**
- Hypothesis commit **`55016fe`** (author-written).
- → M5 code `e554b4a` → thread-check fix `68efdfd` (= `git_commit` of all 55 reported `p4_dim` rows; every row also stores
  `hypothesis_commit = 55016fe` and the Part 4 section SHA-256 `0751800a5d80a568…`) → this journal commit.
- The guard ran first in each run.
- Run 1 (commit `e554b4a`) was superseded by run 2 because of bug M5-2. Its log is kept as `results/metadata/part4_run1_threadcheck_flaw.log`.

**Files created:** `src/synthetic.py` (the Part 4 data generator), `experiments/part4_dimensionality.py`, `experiments/part4_report.py`,
`tests/test_synthetic.py`, `../ai_log/milestone5/{prompts,response}.md`.
**Modified:** `src/timing.py` (INCONCLUSIVE thread check), `tests/test_threads_timing.py` (+1), `run_all.sh` (Part 4 stages), `README.md`,
`IMPLEMENTATION_LOG.md`, `../ai_log/ai_statement.md`.
**Outputs (outside git):**
- `results/results.csv` (now 599 rows)
- `results/figures/part4_dimensionality.png` (**the one main figure for Part 4**)
- `results/tables/part4_{recall_by_d,qps_by_d,distance_stats}.{csv,md}`
- `results/metadata/part4_distance_stats.json` (per-d statistics, data fingerprints, seed policy)
- logs `part4.log` (run 2) and `part4_run1_threadcheck_flaw.log`
- `results/quick/` (quick run)

**Design (one property varied: dimensionality d)**

| Held fixed for every d | Varied |
|---|---|
| Generator: x ~ N(0, I_d) float32, L2-normalised (uniform on the unit sphere), "random normalized vectors" as in hypothesis 4.1. Seed policy: `SeedSequence([12345, d]).spawn(2)` → independent base and query streams. n_base = 100,000; n_query = 1,000; Euclidean metric; tie-aware recall@10 against a float64-re-scored exact top-100. Builds: HNSW M=16, efC=200; IVF-Flat nlist=1024 (seeded k-means per d). **Fixed, pre-specified search settings:** efSearch ∈ {16, 32, 64, 128, 256}, nprobe ∈ {1, 4, 16, 64, 256}. Timing: single query, 1 thread, pinned P-core, R = 5 + warm-up | d ∈ {32, 64, **128 (control)**, 256, 512} |

- Settings were **not tuned on the data**. No configuration is selected from the results, so there is no tune/eval split; all 1,000 queries are used.
  This follows hypothesis 4.2 ("fixed, documented search settings") and avoids any selection effect.
- Size: 5 settings × 100,000 vectors. Largest synthetic set = 100,000 (+1,000 queries), well within the assignment's 500K limit.
- Reproducibility: each d's data has a SHA-256 fingerprint. Run 1 and run 2 produced **byte-identical data** and identical recall (max |Δ| = 0.0000 over 55 configurations).

**Commands executed**
```
.venv/Scripts/python -m pytest -q                                          # 100 -> 109 -> 110 passed (bug M5-1)
.venv/Scripts/python -m experiments.part4_dimensionality --quick --dev --repeats 3 ; part4_report --quick   # ~10 s
git commit e554b4a
.venv/Scripts/python -m experiments.part4_dimensionality                   # run 1, 4.8 min (2 thread-check FAILs -> bug M5-2)
python -c "process_time increments over 1 s"                               # [15.625] ms
git commit 68efdfd
.venv/Scripts/python -m experiments.part4_dimensionality                   # run 2 (reported), 4.9 min
.venv/Scripts/python -m experiments.part4_report
sha256sum results/...
```

**Measured results (run 2, commit `68efdfd`; 55 rows reportable; thread_check 38 PASS / 17 INCONCLUSIVE / 0 FAIL; single == batch ids 55/55)**

*Distance concentration (`part4_distance_stats.md`; Euclidean distances; all 10⁸ query–base pairs; medians over queries):*

| d | Mean dist. | Std | CV = std/mean | Median NN dist. | Relative contrast (p10–p90) | d10 / mean | (d11 − d10)/d10 | Ties at rank 10 |
|---|---|---|---|---|---|---|---|---|
| 32 | 1.4085 | 0.1267 | 0.0900 | 0.802 | 1.755 (1.678–1.879) | 0.631 | 0.00286 | 0 |
| 64 | 1.4114 | 0.0890 | 0.0630 | 0.991 | 1.425 (1.389–1.481) | 0.746 | 0.00165 | 0 |
| **128** | 1.4128 | 0.0627 | 0.0444 | 1.121 | 1.261 (1.240–1.295) | 0.824 | 0.00103 | 0 |
| 256 | 1.4135 | 0.0443 | 0.0313 | 1.211 | 1.167 (1.155–1.188) | 0.878 | 0.00070 | 0 |
| 512 | 1.4139 | 0.0313 | 0.0221 | 1.273 | 1.111 (1.104–1.123) | 0.915 | 0.00044 | 0 |

What the table shows:
- The mean distance tends to √2 = 1.4142, as expected for random unit vectors.
- The CV falls by √2 per doubling of d (0.0900 / 0.0221 = 4.07 ≈ √16), i.e. ∝ 1/√d.
- The nearest neighbour moves towards the typical distance: relative contrast 1.755 → 1.111, and d10 reaches 91% of the mean distance at d = 512.

*recall@10 at fixed settings (`part4_recall_by_d.md`; change vs the d = 128 control):*

| Setting | d=32 | d=64 | d=128 | d=256 | d=512 |
|---|---|---|---|---|---|
| HNSW ef=16 | 0.539 | 0.214 | 0.078 | 0.038 | 0.023 |
| HNSW ef=64 | 0.873 | 0.512 | 0.237 | 0.120 | 0.074 |
| HNSW ef=256 | 0.996 (+0.419) | 0.873 (+0.296) | **0.576** | 0.351 (−0.225) | 0.231 (−0.346) |
| IVF nprobe=16 | 0.455 | 0.268 | 0.163 | 0.115 | 0.081 |
| IVF nprobe=64 | 0.775 | 0.541 | 0.378 | 0.290 | 0.214 |
| IVF nprobe=256 | 0.978 (+0.242) | 0.870 (+0.133) | **0.736** | 0.630 (−0.106) | 0.529 (−0.207) |
| Flat (exact) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

All 10 HNSW and IVF series decrease monotonically with d. Full table, including ef=32/128 and nprobe=1/4, in `results/tables/part4_recall_by_d.md`.

*Equal-QPS comparison (IVF-Flat recall interpolated in log-QPS at each HNSW point's QPS; run 2):*

| d | HNSW vs IVF-Flat recall at the same QPS |
|---|---|
| 32 | **HNSW higher at all 4 points**: e.g. 7,241 QPS: 0.967 vs 0.935; 14,341 QPS: 0.873 vs 0.817 |
| 64 | **IVF-Flat higher at all 4 points** (gaps 0.036–0.064): 4,665 QPS: 0.705 vs 0.741 |
| 128 | IVF-Flat higher at all 5: 1,591 QPS: 0.576 vs 0.686; 3,101 QPS: 0.385 vs 0.502 |
| 256 | IVF-Flat higher at all 5: 1,048 QPS: 0.351 vs 0.510 |
| 512 | IVF-Flat higher at all 5: 853 QPS: 0.231 vs 0.321 |

*Throughput at fixed settings (`part4_qps_by_d.md`):* every index slows with d, because the cost per distance grows with d.

| | d=32 → d=512 | Factor |
|---|---|---|
| Exact | 2,305 → 100 QPS | 23× (d grows 16×) |
| HNSW ef=256 | 3,348 → 853 QPS | 3.9× |
| IVF nprobe=256 | 5,661 → 350 QPS | 16.2× |

**Hypotheses vs results** (hypotheses written by the author in `55016fe`; verdicts are the reading of the measured data, for the author to confirm)

| Hypothesis | Prediction | Measured | Reading |
|---|---|---|---|
| **4.1** distances concentrate with d; recall@10 at a fixed budget decreases for at least some configurations | concentration ↑, recall ↓ | CV 0.090 → 0.022 (∝ 1/√d); relative contrast 1.755 → 1.111; 10/11 gap 0.0029 → 0.0004. **Every one** of the 10 ANN series loses recall monotonically with d. Example: HNSW ef=256 0.996 → 0.231, IVF nprobe=256 0.978 → 0.529 | **Supported**, and more broadly than predicted: all configurations, not just some |
| **4.2** HNSW and IVF-Flat respond differently; their relative recall/throughput may change across d | different sensitivity; relative ranking may change | Relative to the control, HNSW changes more in both directions (ef=256: +73% at d=32, −60% at d=512) than IVF (nprobe=256: +33%, −28%). At equal QPS, the **ranking flips** between d = 32 (HNSW better) and d ≥ 64 (IVF-Flat better at every comparison point) | **Supported**: the response differs in size, and the better index changes with d |

*Mechanism (inference, consistent with the measurements but not tested directly):*
- As d grows, the true neighbours are barely closer than everything else (d10 = 0.82–0.91 × the mean distance for d ≥ 128).
- **HNSW** steers greedily by distance comparisons. When all candidates look about equally far, each step carries little information, so a fixed efSearch beam misses more neighbours. A fixed M = 16 graph also has to cover a space of growing intrinsic dimension, which uniform data has (= d).
- **IVF-Flat** only has to land the query's neighbours in the probed cells. On structureless data the k-means cells are arbitrary, and the 10 neighbours spread over more cells as d grows. It degrades too, but more slowly at a fixed probed fraction, and it computes exact distances inside the probed cells.
- This explains the ranking flip, but **not tested:** graph navigation statistics (hops, distance evaluations per query) were not recorded.

*Implication for report Q3:*
- On SIFT (real, structured, d = 128) HNSW reached 0.99 recall at efSearch ≈ 100. On uniform d = 128 data, the same index reaches only 0.576 at efSearch = 256.
- The benchmark numbers in Part 2 therefore depend on the **intrinsic structure** of the data, not on the nominal dimension.
- On uniform data the "HNSW is fastest" headline from Part 2 **reverses** once d ≥ 64.

**Bugs and anomalies**
1. *Distance statistics used the unit-norm identity ‖q − x‖² = 2 − 2q·x.* After float32 normalisation the norms are only 1 ± 1e-7, so the mean differed from brute force by 4e-9 relative. The new test caught it.
   **Fix:** use the actual norms (‖q‖² + ‖x‖² − 2q·x). Found before any reportable run.
2. *Thread check gave false FAILs on very short passes.* Run 1 flagged 2 rows FAIL (cpu/wall 1.70 and 1.31). Their timed passes lasted 9.2 and 11.8 ms, while
   `time.process_time()` on Windows advances in **15.625 ms** steps (measured), so the ratio is quantisation noise.
   - **Fix:** `68efdfd` reports INCONCLUSIVE for passes under 62.5 ms (4 ticks), with a test.
   - Part 4 was re-run on that commit. 17 rows are INCONCLUSIVE, 0 FAIL.
   - Search is pinned to one logical CPU, so it cannot use more than one core anyway.
   - Effect on earlier milestones: rows with passes under 62.5 ms were labelled PASS. That label was equally uninformative, but all of those runs were pinned to one CPU, so no measurement is affected.
3. *Sustained slowdown in one measurement (kept, flagged).*
   - Exact search at d = 128 measured **262.3 QPS** in run 2 (5 repeats 262.0–263.0) against 391.1 in run 1, and it breaks the smooth trend.
   - The same kind of event as the M4 anomaly: a whole measurement uniformly slow, invisible to within-run repeats.
   - Across all 55 configurations, run 2 / run 1 QPS = 0.67–1.19, mean |change| 4.7%, 7 configurations > 10% (mostly very fast configurations, plus this one).
   - **Reported values = run 2** (the reportable rows).
   - **Robustness of the equal-QPS ranking:** a 10% QPS error moves the interpolated IVF recall by at most 0.022–0.026.
     The IVF advantage is 0.036–0.064 at d = 64 (5–25% relative), 0.056–0.117 at d = 128 (19–72%), 0.039–0.159 at d = 256 (45–102%),
     and 0.016–0.090 at d = 512 (39–75%).
   - So the flip is robust at d = 128 and 256, holds with a small margin at d = 64 (minimum gap 1.4× the timing effect), and
     the single fastest d = 512 point (gap 0.016) is within timing noise.

**Provenance (A6), SHA-256 of outputs at this commit:**
```
38ae45bf357ae37e7ee010a4bdf084721b9c455b87fe55fd737a9f639b69536f  results/results.csv
0f65ba57bb92025e6ec14e079478e319ff30925ffe4f5394273227a9f6da3b86  results/figures/part4_dimensionality.png
14dd75ece75d43fd8a693dad2f6564e6d43f6370f9b0c9c8135d3c378afafba2  results/tables/part4_recall_by_d.csv
8736f05cca39a1741a6f7cfa715c25c02f18b708b14e6fcd3660a00517d75b04  results/tables/part4_qps_by_d.csv
705b6f51e4a15363c7fb9178a564149487c39dc3d053434c03d823f78d2ad514  results/tables/part4_distance_stats.csv
76efc0fc1ecb7915b39ebc95dd3ffe8a31af02059c707034ab832bb586275558  results/metadata/part4_distance_stats.json
```

**What M5 establishes:**
- On seeded uniform unit-sphere data with n = 100K, distances concentrate as 1/√d.
- At fixed search budgets, HNSW and IVF-Flat both lose recall steadily with d.
- HNSW is more sensitive to d, and its advantage at equal QPS reverses above d = 32.
- The data generation is exactly reproducible.

**What it does not establish:**
- The behaviour of *structured* data whose intrinsic dimension differs from d (for example clustered data or real embeddings).
- The cost of keeping recall constant as d grows (settings were fixed by design).
- Other M, efConstruction or nlist values per d.
- Other n.
- IVF-PQ.
- The causal mechanism (graph or cell statistics not recorded).

**Remaining / next:** Milestone 6, the report (≤ 5 pages; Part 4 gets one main figure and ≤ 1 page), the final `run_all.sh` check, and the submission zip.

---

## 6. Algorithms, formulas and implementation decisions

- L2 normalisation: x̂ = x / ‖x‖₂ (rows with ‖x‖ = 0 stay 0).
- Equivalence: ‖q̂ − x̂‖² = 2 − 2 cos(q, x), so ranking is identical to angular ranking.
### Approved decisions (2026-10-16), to be implemented in M2 unless stated otherwise

**A1. CPU timing methodology**
- **Single thread for all search timings:** both `faiss.omp_set_num_threads(1)` **and** FAISS's OpenBLAS thread count = 1 are set before any timed search (see A5 for why both are needed). Both values are read back and recorded.
- **Pinning:** the timing process is pinned to **one logical processor of one P-core**. The core is chosen at runtime from the
  OS-reported mapping (EfficiencyClass = maximum), never from a hardcoded id. Plan: use the first logical processor of the
  *last* P-core, because core 0 typically services more OS interrupts. On this machine that resolves to logical processor 10, but
  the code must derive this, not assume it. After pinning, the affinity is read back. If setting it fails, the mapping cannot
  be read, or the readback does not match, the harness **falls back to unpinned timing** and records `pinned=False` with the reason.
- **Verification in M2 (measured, not assumed):** a short calibration run times the same exact-search workload pinned to a P-core,
  pinned to an E-core and unpinned. The result shows whether the P/E difference is real and whether pinning reduces spread.
- **Repetition:** each configuration gets 1 untimed warm-up pass, then R timed passes (planned R = 5) over the same query set.
  QPS = n_queries / wall-time of one pass, using `time.perf_counter()`. We report the **median** of the R passes, and keep the min, max and all R values in `results.csv`.
- **One query at a time:** the loop calls `index.search(q[i:i+1], k)` for each query. The Python call overhead is the same for every index and is included.
- **Recorded per row:** `pinned`, `logical_cpu`, `efficiency_class`, `omp_threads`, `n_repeats`, `qps_median/min/max`, git commit and timestamp.
- **Limitations:** the SMT sibling of the pinned core (logical 11) may be used by other processes. Turbo frequency and thermals are not
  controlled. Windows can still preempt the thread. Background load is minimised but not eliminated.

**A2. recall@10, distance-based**

Definitions for one query q with k = 10:
- `true_d(i) = ‖q − x_i‖²` is computed by us in **float64** from the stored float32 vectors, for both the reference and the candidate ids.
  Using the same arithmetic for both sides prevents spurious mismatches between FAISS's float32 distances and ours.
- `τ = d*₁₀`, the 10th-smallest true distance from the exact reference (FAISS `IndexFlatL2` results, re-scored in float64).
- Returned id r counts as **correct** if `r ≥ 0`, it is the first occurrence of that id in the returned list, and
  `true_d(r) ≤ τ + tol`, where `tol = rel_tol · max(τ, tiny)` with `rel_tol = 1e-6`.
- **recall@10 = (#correct among the first 10 returned) / 10.** The denominator is always 10, even if the index returns
  fewer than 10 ids or returns `-1` padding.

Why this tolerance:
- Both sides are computed identically in float64 from the same float32 data, so genuine ties give differences around 1e-12 relative.
- SIFT distances are integers and are exact in float64.
- 1e-6 relative is far above round-off, but far below the gap between distinct neighbours. This gap will be **measured** in M2
  (smallest nonzero relative gap between d*₁₀ and d*₁₁).
- `tiny` guards the case τ = 0, where the query duplicates a database vector.

Why duplicate ids are counted once: an index that returned the same correct id 10 times must not get recall 1.0.

Recall is reported both per query and as the mean over the query set.

Unit tests planned for M2:
- perfect result → 1.0;
- disjoint result → 0.0;
- a tied non-reference id at the boundary is counted;
- an id just beyond τ + tol is not counted;
- duplicates are counted once;
- `-1` padding and fewer than 10 results;
- τ = 0 (the query duplicates a database vector);
- different id order gives the same recall;
- a hand-computed partial case (for example 7/10);
- agreement with naive id-intersection recall when there are no ties.

The number of tied queries (`d*₁₀` and `d*₁₁` within tolerance) is **measured** on both datasets and written to
`results/metadata/`. It is not hardcoded. M1's exact-equality counts (SIFT 137, GloVe 1) are only a preliminary measurement.

**A3. Index library: FAISS for everything**

| Index | FAISS class | Build parameters | Search parameter |
|---|---|---|---|
| Exact | `IndexFlatL2` | none | none |
| HNSW | `IndexHNSWFlat(d, M)` | `M`, `hnsw.efConstruction` | `hnsw.efSearch` |
| IVF-Flat | `IndexIVFFlat(IndexFlatL2(d), d, nlist)` | `nlist` | `nprobe` |
| IVF-PQ | `IndexIVFPQ(IndexFlatL2(d), d, nlist, m, nbits=8)` | `nlist`, `m` | `nprobe` (Part 3 adds `IndexRefineFlat` for reranking) |

All indexes use `METRIC_L2` on raw SIFT and on L2-normalised GloVe.

FAISS-specific points that affect comparisons, to document and handle:
1. FAISS L2 returns **squared** distances. Recall does not use FAISS distances (A2), so this does not matter for recall.
2. HNSW `efSearch < k` is effectively raised to k internally, so sweeps start at efSearch ≥ 10.
3. IVF/PQ k-means is seeded (`cp.seed = SEED`). FAISS subsamples training points (max 256 per centroid by default, with a warning).
   The training-set size is recorded.
4. HNSW insertion with several threads is non-deterministic, so the graph and recall can vary slightly between builds. We record
   this and, if time allows, measure build-to-build variance once.
5. Index size is measured identically for every type as `len(faiss.serialize_index(index))` bytes. Bytes per vector = size / n.
   This includes HNSW graph links and IVF ids. It is not process RSS.
6. **Build thread policy:** resolved by A5 below.

**A4. Submission layout:** see section 4. `code/run_all.sh` writes only to `<submission>/results/`. `ai_log/` is maintained by
hand at `<submission>/ai_log/`. Neither is duplicated inside `code/`.

**A5. Thread policy for building and searching** (approved 2026-10-16; controls verified in design revision B)

A planned module `src/threads.py` (M2) provides two phase functions. Each one sets **all three** controls, reads them back,
and returns a record that is written into every result row.

| Phase | Process affinity | `faiss.omp_set_num_threads` | FAISS OpenBLAS (`openblas_set_num_threads` via ctypes) |
|---|---|---|---|
| `build_phase()` | all logical processors with the maximum EfficiencyClass (here 0–11, derived at runtime) | number of those processors (here 12) | same (here 12) |
| `search_phase()` | one P-core logical processor (A1; expected to be 10) | 1 | 1 |

- Every phase switch is **verified**: the affinity, `omp_get_max_threads()` and `openblas_get_num_threads()` are read back.
  A mismatch raises an error. Timing never continues silently with the wrong configuration.
- If the CPU-set API or the OpenBLAS symbols are unavailable (for example a grader on Linux or macOS), the module falls back
  to `os.sched_setaffinity` where it exists, and otherwise to no affinity. The fallback is recorded in the row. On a non-hybrid
  CPU, "P-cores" means all cores.
- **Each timed search region's cpu/wall ratio is also recorded.** A row whose single-threaded search shows cpu/wall above 1.3 is flagged
  `thread_check=FAIL`. It stays in `results.csv` but is excluded from reported numbers, as evidence that the timing was contaminated.

How the policy applies to each index type (build = `train` + `add`, timed with `perf_counter` under `build_phase()`):

| Index | Build stages and which pool they use | Search (always `search_phase()`) |
|---|---|---|
| Exact `IndexFlatL2` | `add` only copies vectors (about 0 s; recorded, but not a meaningful build time) | single query: brute-force scan (OpenMP, so omp = 1 matters). Batch: BLAS matrix product (OpenBLAS, so blas = 1 matters) |
| HNSW `IndexHNSWFlat` | `add` builds the graph, parallel with OpenMP (5.1× on the probe). Multi-threaded insertion order makes the graph non-deterministic | graph search, single query |
| IVF-Flat | `train`: k-means, whose distance steps use BLAS and OpenMP. `add`: assign each vector to its nearest centroid (BLAS/OpenMP) | coarse quantizer + scan of `nprobe` lists, single query |
| IVF-PQ | as IVF-Flat, plus PQ codebook k-means per sub-vector and encoding of every vector. Speed-up unknown at 1M (probe: none at 60K) | as IVF-Flat, with lookup-table distances |
| IVF-PQ + refine (Part 3) | IVF-PQ build + storing the raw vectors (`IndexRefineFlat`) | IVF-PQ top-100, then exact re-ranking |

- Single-query and batched exact timings (Part 1) both run under `search_phase()`, so the reported difference comes from batching itself (BLAS
  matrix products, cache reuse, per-call overhead) and not from threads. A **secondary**, clearly labelled batch row
  with blas = omp = 12 may be added to show how threading combines with batching.
- **Build timing and query timing are always separate.** The phases run in sequence and are recorded in separate columns. No timed search
  starts until `search_phase()` has been verified. The untimed warm-up pass (A1) also absorbs any leftover spinning from the build threads.
- **No concurrent benchmarks:** a lock file `results/.benchmark.lock`, holding the PID and start time, is created by every benchmark driver.
  A second driver refuses to start while it exists. `run_all.sh` runs its stages one after another. The author should avoid heavy
  background work during runs; this is recorded as a limitation, not enforced.
- **Build-scaling check (M3, measured):** for one configuration per index type, measure the build at 1 and 12 threads once.
  This shows how much of the build time comes from parallelism. It is reported, not used to choose between indexes.
- Limitations: the 12 processors are 6 cores × 2 SMT threads, so the speed-up is bounded by about 6 physical cores plus the SMT gain.
  Build times are comparable only under this policy. HNSW build-to-build variance comes from parallel insertion.

**A6. Hypothesis provenance and result tracking** (approved 2026-10-16; implemented with Parts 3/4, guard tests in M2 or M4)

A planned module `src/provenance.py` provides:

`require_committed_hypothesis(section)` is called by the Part 3 and Part 4 drivers **before any data generation, index build
or query**, and before both tuning and final runs. It refuses to start (exit code 3, nothing written) unless **all** of these hold:
1. `git` is available, the code is in a git repository, and `hypothesis.md` is tracked (`git ls-files --error-unmatch hypothesis.md`).
2. `hypothesis.md` has no staged or unstaged changes (`git status --porcelain -- hypothesis.md` is empty).
3. The working-copy bytes equal `git show HEAD:hypothesis.md`, after normalising CRLF to LF. `.gitattributes` forces LF, so this guards against tools that change line endings.
4. The requested section (`## Part 3` or `## Part 4`) exists, and with headings stripped it is non-empty, contains none of the template
   placeholder text (`_Not yet written`), and has at least 30 words of content.
5. The hypothesis commit (`git log -1 --format=%H -- hypothesis.md`) is an ancestor of HEAD (`git merge-base --is-ancestor`).

It returns `hypothesis_commit`, the SHA-256 of the section text, and the pre-run `HEAD`.

Final (reportable) Part 3/4 runs also require **no uncommitted changes to tracked source files**, so `git_commit` truly
identifies the code that produced the numbers. A `--dev` flag allows dirty-tree runs for debugging. Their rows are marked
`git_dirty=True, reportable=False` and are never used in the report. Untracked files outside `code/` (data, results) do not count.

**Result-row provenance:** every row of `results/results.csv` (all parts, not only 3/4) contains:

| Group | Columns |
|---|---|
| Provenance | `run_id` (UUID), `experiment_id` (e.g. `p3_compression`, `p4_dim`), `timestamp_utc` (pre-run), `git_commit` (pre-run HEAD), `git_dirty`, `reportable`, `hypothesis_commit` + `hypothesis_sha256` (Parts 3/4; empty otherwise) |
| Input | `dataset_id` (e.g. `sift`, `glove`, `synth_d128_seed12345`), `n_base`, `query_split` (`tune`/`eval`), `n_queries`, `seed` |
| Configuration | `index_type`, `build_params` (JSON), `search_params` (JSON) |
| Threads | `build_affinity`, `build_omp`, `build_blas`, `search_affinity`, `search_omp`, `search_blas`, `pinned`, `pin_fallback_reason`, `search_cpu_wall`, `thread_check` |
| Metrics | `recall_at_10`, `qps_median`, `qps_min`, `qps_max`, `n_repeats`, `build_time_s`, `index_size_bytes`, `bytes_per_vector` |
| Outcome | `status` (`ok` / `failed` / `timeout`), `error` |

Failed configurations get a row with `status=failed` and the error message. They are never silently skipped.

**After each Part 3/4 run** (and each M2/M3 benchmark stage):
1. Append to this journal: run ids, the measured outcome, failures, interpretation, output paths, **and the SHA-256 of `results.csv`
   and of each figure produced.**
2. Commit the journal update. The commit message names the experiment and the hypothesis commit.

Resulting history for each experiment: `H` (hypothesis commit) → … → `HEAD` at run time (= the `git_commit` in the rows) → `J`
(journal summary commit). `H` must be an ancestor of `J`.

**What this evidence does and does not prove**
- It **does** show order within the repository: the hypothesis text existed in a commit that is an ancestor of the commit recorded in the
  result rows and of the journal commit. The file hashes in `J` make later edits to `results.csv` or the figures **detectable**.
- It does **not** prove the timing to an outsider. Git commit dates and our UTC timestamps come from the local clock and can be
  set to anything. Commits can be rewritten. Results outside git could have been produced before `H` and timestamped later.
  Nothing here is tamper-proof.
- **Stronger, still not cryptographic:** push `H` to the private GitHub repo *before* running. GitHub records when it received the push,
  independently of this machine's clock. Recommended step, but only the author decides when to push.

The journal commits in this workflow are made by Claude Code as part of the approved procedure. Pushes are made only when the author asks.

## 7. Hypotheses and commit identifiers

| Experiment | Hypothesis commit | Results commit | Status |
|---|---|---|---|
| Part 3: compression | **`55016fe`** (2026-10-17 14:43 IST, pushed to GitHub the same minute) | rows from `b230f71` (run 2026-10-22 08:43–09:05 UTC); journal summary in the M4 journal commit | **done**: guard passed; all 158 rows store hypothesis_commit `55016fe` + section SHA-256 `f6912f61…` |
| Part 4: dimensionality | **`55016fe`** (same commit) | rows from `68efdfd` (run 2026-10-22; run 1 on `e554b4a` superseded); journal summary in the M5 journal commit | **done**: guard passed in both runs; all 55 rows store hypothesis_commit `55016fe` + section SHA-256 `0751800a…` |

Mechanism: decision **A6** (section 6). For each experiment, this table will hold the hypothesis commit `H`, the pre-run HEAD recorded in
the rows, the journal commit `J`, and whether `H` was pushed to GitHub before the run (with the push time shown on GitHub).

## 8. Configuration parameters

| Parameter | Value | Reason |
|---|---|---|
| `SEED` | 12345 | single global seed (spot-check sampling now; k-means sampling and synthetic data later) |
| `N_TUNE_QUERIES` | 5000 | assignment |
| `K` | 10 | recall@10 |
| `chunk_rows` | 100,000 | keeps streaming validation under ~60 MB for these dimensionalities |
| HNSW efConstruction | 200 (fixed) | common default; varying it was out of scope |
| HNSW M | {8, 16, 32} | low / medium / high graph degree |
| IVF nlist | {1024, 4096, 16384} (IVF-PQ: {1024, 4096}) | about √n to 16√n for n ≈ 1M |
| IVF-PQ m, nbits | SIFT {16, 32, 64}, GloVe {10, 25, 50}; nbits 8 | m must divide d; 8 bits = 256 centroids per sub-vector (FAISS default) |
| k-means seed | 12345 (`cp.seed`) | reproducible centroids, shared per nlist |
| Part 2 tune timing | tune[0:1000], R = 3 + warm-up | cost; used only for selection |
| Part 2 eval timing | eval[0:1000], R = 5 + warm-up | A1; reported numbers |
| Sweep stop rules | recall ≥ 0.995; QPS < exact QPS; plateau (2 × gain < 0.002); nprobe ≤ nlist | saves time; logged |
| Part 3 nlist / m / k_factor | 4096 / {8, 16, 32, 64} / 10 (= top 100) | best plain IVF-PQ nlist in M3; m doubles from 8 B to 64 B codes; top 100 per the assignment |
| Part 3 recall targets | 0.50, 0.80, 0.90, 0.95 | lower targets let plain IVF-PQ be compared at all |
| Part 4 data | Gaussian → L2-normalised, n = 100,000, 1,000 queries, `SeedSequence([12345, d]).spawn(2)` | hypothesis 4.1 "random normalized vectors"; one stream per d and per role |
| Part 4 settings | HNSW M=16 efC=200, ef ∈ {16,32,64,128,256}; IVF-Flat nlist=1024, nprobe ∈ {1,4,16,64,256} | fixed and pre-specified (hypothesis 4.2); never tuned on the data |
| `MIN_CHECK_WALL_S` | 0.0625 s | 4 ticks of the 15.625 ms Windows `process_time` clock; shorter passes → INCONCLUSIVE |
| `N_SPOT_CHECK` | 10 queries | cheap ground-truth sanity check. The full check is Part 1 |
| `ANN_DATA_DIR` | default `../data` | portability |
| `ANN_RESULTS_DIR` | default `../results` | submission layout (A4) |
| `rel_tol` (recall) | 1e-6 | A2; SIFT nearest non-tie gap 1.34e-5 (13× above). GloVe has a continuum of gaps (sensitivity table in M2) |
| timing repeats R | 5, plus 1 warm-up | A1; median reported |
| FAISS search threads (OpenMP / OpenBLAS) | 1 / 1 | assignment + A1 + A5; both pools verified to matter |
| `distance_compute_blas_threshold` | FAISS default 128000; set to 20 for the primary batch row and the untimed reference | M2 finding; see the M2 entry |
| Part 1 timed queries | single/default-path: first 500 eval; BLAS path: all 5,000 eval | about 13 s per pass at 38 QPS; recall computed on exactly the timed queries |
| Part 1 repeats | primary 5 (+1 warm-up), secondary 3 (+1) | A1 |
| `--quick` base size | first 100,000 vectors; outputs in `results/quick/` | assignment suggestion |
| pinned search logical CPU | derived at runtime = 10 (verified in `environment.json`) | A1; diagnostic: all P-core CPUs within 2.4% |
| build affinity / OpenMP / OpenBLAS | all P-core logical CPUs (expected 0–11) / 12 / 12 | A5 |
| `thread_check` cpu/wall threshold | 1.3 | A5; single-threaded probe runs measured 0.9–1.2 |
| hypothesis minimum length | 30 words, no template text | A6 |
| guard refusal exit code | 3 | A6; distinct from validation codes 1/2 |

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
| test_results_dir_defaults_to_top_level_outside_repo | (added in the design revision) results go to `<submission>/results`, outside the repo |

Design revision, **measured** 2026-10-16: `19 passed`.

Milestone 2, **measured** 2026-10-17: **`55 passed in 6.00s`** (36 new tests; one test added during M2 after bug 1).

Milestone 3, **measured** 2026-10-17: **`90 passed`**.
- +1 guard test for numbered headings (`468fb36`).
- +12 in `test_part2.py`: Pareto front removes dominated points; equal recall keeps the faster point; target selection is the fastest meeting the target, or none;
  the target pick always lies on the front (200 random points); all 3 builders produce working indexes whose search parameter raises recall
  (IVF-Flat and HNSW > 0.95 at high settings); shared k-means gives results identical to standalone training and charges its time; PQ m must divide d;
  IVF-PQ is smaller than IVF-Flat; BuildSpec name round-trip; custom results key keeps other index types.
- +21 compile tests (`test_scripts_compile.py`), one per `src/` and `experiments/` file. Added after bug M3-2.
- +1 from the new `experiments/build_scaling.py` file being picked up by the compile test.

Real-data validation in M3: single-query ids equal batch ids on every timed subset (367/367 rows); thread_check PASS on 367/367.

Milestone 4, **measured** 2026-10-22: **`100 passed`**.
- +3 `pool_recall` tests: neighbours at the end of the pool are counted (and recall@10 of the same list is 0); partial counts and the cap at 1 with ties; duplicates and padding ignored.
- +6 re-ranking tests: the wrapper shares the base index and stores the raw vectors; re-ranked recall equals pool recall and is never below plain (nprobe 1/4/32); wrong-shaped vectors are rejected before FAISS reads them.
- +1 compile test (the new scripts are picked up automatically).

Real-data validation in M4: re-ranked recall = pool recall in 78/78 rows; ids match on 158/158; thread_check PASS on 158/158.

Milestone 5, **measured** 2026-10-22: **`110 passed`**.
- +6 in `test_synthetic.py`: deterministic and seed-dependent generation; separate streams per d and per role (queries unchanged by n_base);
  unit norm, float32, contiguous, id string (d = 32/128/512); distance statistics equal brute force (mean to 1e-9 relative) and concentrate with d (mean → √2, CV and relative contrast fall).
- +1 thread-check test (INCONCLUSIVE below 62.5 ms; FAIL/PASS above).
- +3 compile tests (`src/synthetic.py` and the two Part 4 scripts).

Real-data validation in M5: byte-identical data and recall in two runs; ids match on 55/55; 0 thread-check FAILs.

| Test file | Tests | What they verify |
|---|---|---|
| `test_recall.py` | 16 | perfect = 1; disjoint = 0; order-invariant; tied non-reference id counts; inside vs outside the tolerance; hand-computed 0.7; duplicates once; −1 padding and short lists keep denominator 10; −1 never counts; τ = 0; equals naive id-intersection without ties; per-row independence; shape errors; float64 distances with padding; deterministic tie order of the reference; tie statistics including tolerance sensitivity |
| `test_threads_timing.py` | 7 | topology covers all CPUs; search phase = 1 pinned CPU, OpenMP = OpenBLAS = 1 (read back); build phase = all fast CPUs; FAISS OpenBLAS found on Windows; single and batch timing give identical ids, each point is its own nearest neighbour; lock blocks a second holder and cleans up; stale lock replaced |
| `test_provenance.py` | 11 | guard refuses template, unstaged, staged-only, too short, missing section, untracked file, non-UTF-8 file, dirty source (unless dev); accepts a committed hypothesis and returns its commit; git_info ignores untracked files; results.csv replacement idempotent; unknown columns rejected |
| `test_data_loader.py` | +1 | subset load keeps the prefix and drops the stale ground truth |

Real-data validation in M2: the ground-truth agreement checks in the M2 entry (all 10,000 queries, both datasets).

Real-data validation, **measured** 2026-10-16: SIFT PASSED, GloVe PASSED (all checks; details in section 3).

## 11. Bugs encountered

None in Milestone 1.

**Milestone 5**
1. *Distance statistics assumed exactly unit norms* (2 − 2q·x), giving a 4e-9 relative error. Fixed to use the actual norms; caught by the new brute-force test before any reportable run.
2. *Thread check gave false FAILs for passes under one CPU-clock tick* (`process_time` resolution 15.625 ms on Windows, measured). Fixed in `68efdfd` (INCONCLUSIVE); Part 4 re-run.
3. *A diagnostic one-liner hung.* It waited for 6 distinct `process_time` increments, but every increment is identical. Stopped (TaskStop) and rewritten as a fixed 1-second loop. No effect on the results.

**Milestone 4**
1. *Same duplicate-keyword bug as M3-1* (`search_mode`, now in the eval-row copy in `part3_compression.py`). Caught by the first `--dev` quick run, before any reportable run.
   Lesson repeated: `TimingResult.as_row()` owns `search_mode`.
2. *Memory-safety hazard found by a test (fixed before any real run).* `faiss.IndexRefineFlat(base, swig_ptr(x))` copies `base.ntotal` rows from a raw pointer
   without bounds checking. With a too-short array it read past the end, with no error. `add_exact_refine` now validates `x.shape == (ntotal, d)` before the call.
   The test checks three bad shapes.

**Milestone 3**
1. *Duplicate keyword (`search_mode`) in `part2_ann.py`.* `TimingResult.as_row()` already sets it. Caught by the first quick run (TypeError) and fixed before any reportable run.
2. *Syntax error committed (`e341426`).* A `sed` replacement inserted a literal newline into an f-string in `part2_report.py`. The tests did not
   import that script, so they passed.
   - **Fix:** `78e0e5f`, plus `tests/test_scripts_compile.py`, which compiles every source file.
   - Found before any reportable run because I checked the grep output.
3. *Misleading table column.* `max_eval_recall` in `part2_max_recall` was the highest eval recall among the *evaluated* configurations of a build, not the
   eval recall of the build's best configuration. For example, GloVe HNSW M=8 showed 0.38 next to a tune maximum of 0.866.
   - **Fix:** `77f89d2`, which reports `eval_recall_same_config` (or "not on tune Pareto front").
   - Measured rows were unaffected; only the derived table changed.
4. *Not a code bug:* the first `pip install matplotlib` stalled on the network for over 5 minutes (about 0 CPU). It was stopped (only that process) and retried with `--timeout 30`.

**Milestone 2**
1. *Test encoding bug.* Three provenance tests failed because `Path.write_text` without `encoding=` uses the Windows code page.
   The "—" in the template became byte 0x97, and the guard then crashed with `UnicodeDecodeError`.
   - **Fix:** the tests write UTF-8 explicitly.
   - **Code hardening:** the guard now refuses a non-UTF-8 `hypothesis.md` with a clear message instead of crashing. A test was added.
   - Root cause: platform default encoding.
2. *Not a bug, but a wrong assumption caught by measurement:* FAISS 1.15.1 uses BLAS for exact search only at ≥ 128,000 queries.
   The plan assumed batch = BLAS. Found by the 20-query sizing run (batch ≈ single) and diagnosed with an explicit threshold sweep.
3. *Process bug: first full Part 1 run not reportable.* While run 1 was in progress I edited the tracked `run_all.sh`.
   `run_metadata()` for the dataset stage ran after that edit, so the 10 `p1_exact` rows were correctly marked `git_dirty=True, reportable=False`
   (the calibration rows, recorded earlier, were clean). The guard worked as designed.
   - **Fix:** commit `e098ca9`, then a full re-run (run 2) with **no edits to tracked files during the run**. Run 2's rows replaced run 1's.
   - Run 1's log is kept as `results/metadata/part1_run1_dirty_tree.log`.
   - Lesson: never edit tracked files while a reportable benchmark is running.

## 12. Benchmark measurements

Part 1 (M2): see the Milestone 2 entry in section 5. Source: `results/results.csv` rows `p1_exact` (10) and `p1_calibration` (3), commit `e098ca9`.

Part 4 (M5): see the Milestone 5 entry. Source: `p4_dim` (55 rows, commit `68efdfd`, hypothesis `55016fe`) and `results/metadata/part4_distance_stats.json`.

Part 3 (M4): see the Milestone 4 entry. Source: `p3_compression` (158 rows: 79 tune + 79 eval, commit `b230f71`, hypothesis `55016fe`).

Part 2 (M3): see the Milestone 3 entry. Source: `p2_ann` (367 rows: 252 tune + 115 eval, commit `78e0e5f`) and `p2_build_scaling` (6 rows, `398f56f`).

## 13. Figures and tables

Part 1 needs no figure. Its tables are in the M2 entry, derived from `results.csv` and `results/metadata/part1_*_reference.json`.

| Output | What it shows | Generated by |
|---|---|---|
| `results/figures/part2_sift_recall_vs_qps.png` | SIFT: eval recall@10 vs single-thread QPS (log scale) of each index's tune-selected Pareto configurations, plus the exact baseline star, with 0.90/0.95 guides | `experiments/part2_report.py` |
| `results/figures/part2_glove_recall_vs_qps.png` | same for GloVe | same |
| `results/tables/part2_qps_at_recall.{md,csv}` | QPS at recall 0.90/0.95: config chosen on tune, recall and QPS on eval, whether eval meets the target | same |
| `results/tables/part2_build_cost_at_095.{md,csv}` | build time, size, bytes/vector of the 0.95 configuration | same |
| `results/tables/part2_max_recall.{md,csv}` | best tune recall per build, the setting where it occurs, eval recall of that setting | same |
| `results/figures/part3_compression.png` | **main Part 3 figure**: (a) eval recall vs QPS per m, plain (dashed) vs re-ranked (solid); (b) bytes/vector vs best recall, with pool recall 10@100 | `experiments/part3_report.py` |
| `results/tables/part3_by_m.{md,csv}` | per m: memory, build time, best recall (plain / re-ranked / pool), QPS at 0.50/0.80/0.90/0.95 for both variants | same |
| `results/tables/part3_rerank_cost.{md,csv}` | per evaluated re-ranked configuration: refine QPS, candidate (k = 100) QPS, re-rank ms per query and share of latency | same |
| `results/figures/part4_dimensionality.png` | **main Part 4 figure**: (a) recall@10 vs d at each fixed setting (HNSW solid, IVF-Flat dashed); (b) single-thread QPS vs d with the exact baseline; (c) distance concentration vs d (relative contrast − 1 with p10–p90 band, CV, 10/11 gap) | `experiments/part4_report.py` |
| `results/tables/part4_recall_by_d.{md,csv}` | recall per setting and d, with the change vs the d = 128 control | same |
| `results/tables/part4_qps_by_d.{md,csv}` | QPS per setting and d | same |
| `results/tables/part4_distance_stats.{md,csv}` | distance statistics, ties and data fingerprint per d | same |

## 14. Limitations, assumptions, unresolved issues

- The expected GloVe train count (1,183,514) and the cosine-distance encoding of stored distances come from the ann-benchmarks conventions. The validator will check them.
- Hybrid P/E-core CPU: E-cores measured about 25% slower and unpinned runs vary more (7% vs 1% spread). All reported single-thread timings are pinned to a P-core.
- **Run-to-run variation of about 5%** (SIFT single 36.3 → 38.1 QPS between runs 1 and 2) exceeds within-run spread (1–3%). Do not interpret differences under about 5% between separately run configurations.
- GloVe tie count depends on the tolerance (0–195 queries for 0–1e-4). At 1e-6 only 2 queries are affected.
- Single-query timings use 500 of the 5,000 eval queries (cost). BLAS-path rows use all 5,000.
- Part 2: QPS is timed on 1,000 of 5,000 queries per split; recall uses all 5,000. Tune QPS used R = 3.
- Part 2 sweeps for GloVe HNSW M ≤ 16 and IVF-Flat nlist ≥ 4096 ended at the end of the sweep list (1024), so those maxima are lower bounds, not ceilings.
- The SIFT tune half is harder than the eval half (eval recall higher on all 47 selected configurations, mean +0.016). Tune selection is slightly conservative on SIFT.
- Fixed efConstruction (200) and nbits (8); no OPQ. Different values could change the rankings at the margins (not tested).
- Part 3: one unexplained slow eval measurement (plain m = 64, nprobe 4: 3,357 vs 6,472 QPS on tune) is kept and flagged. The re-rank ms is a difference of two medians and is unreliable when the total latency ≫ 0.02 ms.
- Part 4 uses uniform (structureless) data, so its intrinsic dimension equals d. Structured data was not tested. Settings are fixed by design, so the cost of *constant* recall vs d was not measured.
- Run-to-run QPS variation measured on identical data (Part 4, 55 configurations): mean |change| 4.7%, range 0.67–1.19×. One sustained slowdown (exact, d = 128: 262 vs 391 QPS) is kept and flagged.
- Part 3 keeps the re-rank vectors in RAM as float32 (+512 B/vector). Disk-resident or compressed refine stores were not tested.
- Build times are with 12 P-core threads. At 1 thread HNSW is about 5.5× slower and the IVF types about 1.75× slower (scaling check), so build-time comparisons depend on the thread policy.
- The memory-bandwidth explanation of single-query speed is inferred (about 19 GB/s implied), not measured with a bandwidth benchmark. The reason 12-thread BLAS scales only 2× is not determined.
- `results/` and `ai_log/` are not version-controlled (A4). Their history is not recoverable from git, so the hypothesis-ordering evidence relies on the mechanism in section 7.
- FAISS-specific behaviours listed in A3 (HNSW build non-determinism, k-means subsampling, efSearch ≥ k).
- Threading (design revision B): OpenMP and FAISS's OpenBLAS are independent pools, and FAISS's OpenBLAS defaults to 20 threads. Busy-waiting
  OpenMP threads inflate cpu/wall, so cpu/wall is used only as a single-thread check. IVF build speed-up at 1M is unmeasured.
- Probes and threading controls are Windows-specific (CPU-set API, `.dll` names). Other operating systems use the documented fallbacks, which are untested.
- Provenance (A6) is procedural evidence, not tamper-proof. Clocks and git dates can be set arbitrarily.
- Only ~7 GB of RAM was free at inspection time. Loading one dataset (~0.5 GB) plus an index is fine. Plan to process one dataset at a time.
- `run_all.sh` covers stages up to Part 1. `--quick` uses the first 100K base vectors. The stored ground truth is dropped there, and τ comes from our exact search (implemented in M2).
- The 16 MB-per-dataset reference caches live in `data/cache/`, which is not submitted. `run_all.sh` regenerates them.
- The global Python environment is not used. Graders should install from `requirements.txt`.
- The SIMD level of the FAISS wheel is not recorded yet.

## 15. Decisions / actions required from the user

1. ~~Download the datasets~~. Done 2026-10-16, both validated.
2. ~~Layout~~. Resolved by A4: `results/` and `ai_log/` are at the top level, with no copies.
3. ~~First commit~~. Done (`a17ac94`, `0012673`), pushed to the private GitHub repo.
4. **Save the session transcript** into `<submission>/ai_log/transcripts/` (see `ai_log/ai_statement.md`).
5. ~~P-core pinning~~. Resolved by A1. ~~hnswlib~~. Resolved by A3 (FAISS only).
6. ~~Review and commit the design revision~~. Done (`d526e1e`).
7. ~~Build thread policy~~. Approved and specified as A5, controls verified (design revision B).
8. ~~Hypothesis-ordering mechanism~~. Approved and specified as A6.
9. ~~Review revision B~~. Done (`d526e1e`).
10. ~~Secondary 12-thread batch row~~. Included as labelled secondary rows (no explicit answer was received; the recommendation was applied). Easy to drop if you prefer.
12. ~~Review Milestone 2~~. Approved ("approved. start milestone 3").
13. **Copy the session transcripts** into `ai_log/transcripts/`: session `9d5f9e15-…` (M1–M2) and session `815a2c47-…` (hypothesis commit + M3) when it ends.
14. ~~Review Milestone 3~~. Approved ("approved. commit everything, then start milestone 4").
15. ~~Part 3 design~~. Applied as proposed (nlist 4096, m ∈ {8,16,32,64}, k_factor 10).
16. ~~Review Milestone 4~~. Approved ("start milestone 5"). The Part 3 verdicts were not explicitly confirmed; please confirm them in the report.
18. **Review Milestone 5**: (a) the 4.1/4.2 verdicts (yours to confirm); (b) the fixed-settings design (no tuning, so no split); (c) the robustness caveat on the equal-QPS flip at d = 64 and 512.
17. **Due 25 Oct.** Remaining: Part 4 (M5), report (M6), final `run_all.sh` check, zip. Copy the session transcripts into `ai_log/transcripts/`.
11. **Before each Part 3/4 run:** write the hypothesis yourself, commit it, and (recommended) push it to GitHub.

## 16. Report preparation map

| Report item | Evidence needed | Source (planned) |
|---|---|---|
| Q1 fastest index at recall 0.95, per dataset | HNSW on both: SIFT 6,783 QPS (3.97× IVF-Flat), GloVe 762.5 (1.85×); GloVe 8.9× slower for HNSW; IVF-PQ never reaches 0.95 | M3 ✔ |
| Q2 build time / memory; what PQ saves and loses | M3: HNSW 47–135 s, +80.6–272 B/vec of links; IVF dominated by k-means; PQ 6.9–22.5× smaller but capped at 0.88/0.87. M4: PQ = m + 10.3 B/vec; gives up recall (0.41–0.89, set by m); exact re-ranking recovers it to the pool bound (0.89–0.999) for +512 B/vec and about 0.02 ms per query | M3 ✔, M4 ✔ |
| Q3 Part 3/4 findings; when headline numbers mislead | M4: 3.1–3.3 supported; 'PQ saves memory' misleads once re-ranking needs raw vectors in RAM. M5: 4.1–4.2 supported; on uniform data HNSW loses its equal-QPS lead above d = 32, and at d = 128 reaches only 0.58 recall at ef = 256 against 0.99 at ef ≈ 100 on SIFT, so the headline numbers depend on intrinsic structure. M3: GloVe vs SIFT gap, sweep-limited maxima | M4 ✔, M5 ✔ |
| Q4 150K vs 10M vectors, 100 ms, 16 GB | exact single-query latency: 26.3 ms (SIFT 1M), 25.0 ms (GloVe 1.18M), scan about 19 GB/s → linear in n·d (assumption to check). Flat 512 / 400 B/vector → 10M SIFT = 5.1 GB raw | M2 ✔ (measured part), M3 |
| Q5 two ways the benchmark could mislead | M2: P vs E core (−25%), unpinned variance (7%), run-to-run 5%, FAISS default batch path, tie handling (65 SIFT queries). M3: SIFT query halves differ in difficulty (+0.016 recall); tune/eval QPS differ by up to 34%; build times depend on threads (HNSW 5.45×). M4/M5: sustained slowdowns invisible to within-run repeats (2 cases, −33% to −48%); run-to-run on identical data 4.7% mean; cpu/wall checks need passes ≥ 62.5 ms on Windows | M2 ✔, M3 ✔, M5 ✔ |
| Machine specs / versions | `results/metadata/environment.json` | M1 ✔ |
