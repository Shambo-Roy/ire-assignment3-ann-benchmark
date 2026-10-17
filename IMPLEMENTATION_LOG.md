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
| T2 | Build: multi-threaded on the verified P-core logical processors. OpenMP and OpenBLAS both set and verified. Build timing kept separate from query timing. Build thread configuration recorded (A5) | IN PROGRESS: `build_phase()` implemented, tested and used for Flat; ANN builds in M3 | `src/threads.py` |
| T3 | No concurrent benchmarks (lock file, sequential `run_all.sh`) (A5) | COMPLETED (M2) | `benchmark_lock`, 2 tests |
| G1 | Hypothesis guard: refuse Part 3/4 if the hypothesis is missing, empty, template-only or uncommitted (A6) | IMPLEMENTED + TESTED (M2, 9 tests); used from M4 | `src/provenance.py` |
| G2 | Provenance columns in every result row: commit, UTC time, experiment id, configuration, seed, input id, metrics (A6) | COMPLETED (M2 schema, used by Part 1) | `src/results_io.py` |
| G3 | After each run: journal summary with output hashes, then commit (A6) | IN USE (M2 entry has hashes) | section 5, M2 |
| P2.1 | HNSW, IVF-Flat, IVF-PQ on both datasets: several build settings, search-parameter sweep up to recall ≥ 0.95 | PLANNED (M3) | |
| P2.2 | QPS measured one query at a time, single thread, median of repeats | HARNESS COMPLETED (M2); used for ANN in M3 | `src/timing.py` |
| P2.3 | Recall-vs-QPS plot per dataset (Pareto front per index + exact) | PLANNED | |
| P2.4 | Table: QPS at recall 0.90 / 0.95 | PLANNED | |
| P2.5 | Table: build time, index size, bytes/vector at recall 0.95 | PLANNED | |
| P2.6 | Explain any recall ceiling | PLANNED | |
| P3 | Compression on SIFT: vary PQ m; IVF-PQ vs IVF-PQ + exact rerank of top 100; recall, speed, memory; hypothesis committed first | PLANNED | |
| P4 | Own synthetic data: dimensionality d ∈ {32, 64, 128 (control), 256, 512}, ≤ 500K vectors, seeded, ≥ 2 indexes, hypothesis committed first, 1 figure, ≤ 1 page | PLANNED | |
| P5.1 | Report ≤ 5 pages that answers Q1–Q5 with mechanisms | PLANNED | |
| P5.2 | Machine specs + library versions | IN PROGRESS | `results/metadata/environment.json` |
| P5.3 | `README.md`, `requirements.txt`, `hypothesis.md` | IN PROGRESS | files exist |
| P5.4 | `bash run_all.sh` (full) and `--quick` (under 15 min on a laptop), run from `code/` and writing to the top-level `results/` | IN PROGRESS: stages up to Part 1. Quick Part 1 took 33 s | `run_all.sh` |
| P5.5 | `results/results.csv` (one row per run behind any reported number), `results/figures/`, at the top level | IN PROGRESS: 13 Part 1 rows; no figures yet | `../results/results.csv` |
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
| Part 3: compression | not yet | not yet | template in `hypothesis.md`; the **author must write** it |
| Part 4: dimensionality | not yet | not yet | template in `hypothesis.md`; the **author must write** it |

Mechanism: decision **A6** (section 6). For each experiment, this table will hold the hypothesis commit `H`, the pre-run HEAD recorded in
the rows, the journal commit `J`, and whether `H` was pushed to GitHub before the run (with the push time shown on GitHub).

## 8. Configuration parameters

| Parameter | Value | Reason |
|---|---|---|
| `SEED` | 12345 | single global seed (spot-check sampling now; k-means sampling and synthetic data later) |
| `N_TUNE_QUERIES` | 5000 | assignment |
| `K` | 10 | recall@10 |
| `chunk_rows` | 100,000 | keeps streaming validation under ~60 MB for these dimensionalities |
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

## 13. Figures and tables

No figures yet (Part 1 needs none). Part 1 tables (ground-truth agreement, ties, timing, CPU placement) are in the M2 entry and are derived only from `results.csv` and `results/metadata/part1_*_reference.json`.

## 14. Limitations, assumptions, unresolved issues

- The expected GloVe train count (1,183,514) and the cosine-distance encoding of stored distances come from the ann-benchmarks conventions. The validator will check them.
- Hybrid P/E-core CPU: E-cores measured about 25% slower and unpinned runs vary more (7% vs 1% spread). All reported single-thread timings are pinned to a P-core.
- **Run-to-run variation of about 5%** (SIFT single 36.3 → 38.1 QPS between runs 1 and 2) exceeds within-run spread (1–3%). Do not interpret differences under about 5% between separately run configurations.
- GloVe tie count depends on the tolerance (0–195 queries for 0–1e-4). At 1e-6 only 2 queries are affected.
- Single-query timings use 500 of the 5,000 eval queries (cost). BLAS-path rows use all 5,000.
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
12. **Review Milestone 2** (this entry), in particular: (a) the added BLAS-path vs default-path batch rows; (b) 500-query single-query timing; (c) the GloVe tolerance sensitivity. Then approve Milestone 3.
13. **Copy the session transcript** into `ai_log/transcripts/` when this session ends (it now also covers M2).
11. **Before each Part 3/4 run:** write the hypothesis yourself, commit it, and (recommended) push it to GitHub.

## 16. Report preparation map

| Report item | Evidence needed | Source (planned) |
|---|---|---|
| Q1 fastest index at recall 0.95, per dataset | QPS@0.95 table + Pareto plots | M3 results.csv, figures |
| Q2 build time / memory; what PQ saves and loses | build/size table + Part 3 | M3, M4 |
| Q3 Part 3/4 findings; when headline numbers mislead | hypothesis vs results, Part 4 figure | M4, M5 |
| Q4 150K vs 10M vectors, 100 ms, 16 GB | exact single-query latency: 26.3 ms (SIFT 1M), 25.0 ms (GloVe 1.18M), scan about 19 GB/s → linear in n·d (assumption to check). Flat 512 / 400 B/vector → 10M SIFT = 5.1 GB raw | M2 ✔ (measured part), M3 |
| Q5 two ways the benchmark could mislead | measured in M2: P vs E core (−25%), unpinned variance (7%), run-to-run 5%, FAISS default batch path (no speed-up below 128K queries), tie handling (65 SIFT queries would be wrongly penalised by id matching) | M2 ✔, later |
| Machine specs / versions | `results/metadata/environment.json` | M1 ✔ |
