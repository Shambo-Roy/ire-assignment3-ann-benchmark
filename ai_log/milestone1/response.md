# Milestone 1 response (final summary from Claude Code, verbatim)

Tool: Claude Code (desktop app), model Claude Opus 5.5. Date: 2026-10-16.
The intermediate tool calls and outputs are only in the full session transcript (see `ai_log/ai_statement.md`).

---

## Milestone 1: done in code, blocked on data

**Workspace found:** only the assignment PDF. No git repo, no datasets (also searched Downloads, Desktop and Documents),
no code. Global Python 3.12.10 had numpy but **no FAISS, h5py, pytest or hnswlib**.

**Machine (measured):** i5-13500 (14 cores / 20 threads, hybrid P/E cores), 15.68 GB RAM (~7 GB free),
835 GB free disk, Windows 11 Pro. Project venv: numpy 2.5.4, faiss-cpu 1.15.1, h5py 3.16.0, psutil 7.2.2, pytest 9.1.1.

**Built:** `code/` as the git root, with the datasets expected in a sibling `data/` folder outside the repo.
`src/config.py` (paths, registry, seed), `src/data_loader.py` (header-only inspection, streaming NaN/inf/zero-norm stats,
GloVe L2 normalisation, 5,000/5,000 split, loader), `src/environment_info.py`, `src/validate_data.py`
(shape, dtype, metric-attribute, GT-range and GT-sorting checks, tie count at rank 10, 10-query distance spot check),
tests with synthetic ann-benchmarks-format HDF5 files, a partial `run_all.sh`, README, IMPLEMENTATION_LOG,
a hypothesis template and the AI log.

**Tests (measured):** 18 passed, 0 failed (2.19 s). `validate_data` reports MISSING for both datasets (exit 2), as expected.

**Needed from you:** download both datasets into `Assignment 3/data/`; make the first commit; copy the session transcript into
`ai_log/transcripts/`; confirm the layout, where results/ai_log go in the zip, P-core pinning and FAISS-only HNSW.

**Milestone 2 (not started):** recall@10 with tie handling + unit tests, FAISS exact search, a full check against the stored ground truth,
a single-thread timing harness with medians, single-query vs batched timing, and the `results.csv` schema.
