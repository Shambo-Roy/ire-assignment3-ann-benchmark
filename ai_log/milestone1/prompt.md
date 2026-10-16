# Milestone 1 prompt (verbatim, extracted from the Claude Code session transcript)

Tool: Claude Code (desktop app), model Claude Opus 5.5. Date: 2026-10-16.

---

IRE Assignment 3 — Implementation Prompt (Milestone 1)
1. Your role
Act as a senior Python engineer and experimental researcher helping me implement my individual assignment for the Information Retrieval & Extraction (CS4.406) course.
The assignment is titled “Assignment 3: Benchmarking Approximate Nearest Neighbour Search.”
Read the complete assignment PDF in the repository/workspace before making changes. Treat it as the authoritative specification. Do not silently omit, reinterpret, or replace any requirements.
My goal is to build a correct, reproducible benchmarking system for exact nearest-neighbour search, HNSW, IVF-Flat, and IVF-PQ, perform the required experiments, and prepare a report of at most five pages.
Important: Do not implement the entire assignment in this first task. Start by inspecting the workspace, documenting the plan, establishing the project structure, and implementing Milestone 1 only. Wait for my review before proceeding to Milestone 2.
2. Mandatory requirements from the assignment
The implementation must eventually satisfy all of the following:
Data

* Use SIFT (1 million vectors, 128 dimensions) and GloVe-100 (approximately 1.18 million vectors, 100 dimensions).
* Use Euclidean distance for both datasets.
* Normalize GloVe vectors as required by the assignment.
* Each dataset provides 10,000 queries and ground-truth nearest neighbours.
* Use the first 5,000 queries for tuning and the final 5,000 queries for final evaluation.

Part 1: Exact baseline

* Implement exact search using FAISS.
* Write my own recall@10 evaluation function.
* Handle ties at the 10th neighbour correctly.
* Write unit tests for recall@10.
* Compare exact results with the supplied ground truth.
* Measure search performance for individual queries and batched queries.
* Explain the timing difference.

Part 2: ANN benchmarks

* Benchmark HNSW, IVF-Flat and IVF-PQ on both datasets.
* Explore several index-construction settings and search-time parameters.
* Measure single-query-at-a-time QPS using a single thread.
* Repeat timings and report median performance.
* Produce recall@10 versus QPS plots for both datasets.
* Produce tables of QPS at recall 0.90 and 0.95.
* Produce tables of build time, index size and bytes per vector at recall 0.95.
* Explain any failure to reach high recall.

Part 3: Compression experiment on SIFT
My selected experiment is compression.

* Vary the number of PQ sub-vectors.
* Compare IVF-PQ with IVF-PQ followed by exact reranking of the top 100 candidates.
* Compare recall, speed and memory.
* Write and commit a hypothesis before running the experiment.
* Tune on the first 5,000 queries and evaluate final configurations on the last 5,000.

Part 4: My own synthetic-data experiment
My selected experiment is dimensionality.

* Generate seeded synthetic vector data.
* Compare exact search, HNSW and IVF-Flat as supporting evidence.
* Use dimensionalities 32, 64, 128, 256 and 512, with 128 as the unchanged control.
* Initially target 100,000 indexed vectors and 1,000 queries per setting.
* Use the same generation principles, seed policy and evaluation methodology across settings.
* Keep all synthetic datasets within the assignment's 500,000-vector limit.
* Change one property at a time and commit the hypothesis before generating experiment results.
* Produce one main figure and keep the Part 4 report discussion within one page.

These are proposed experiment settings, not permission to ignore any mandatory assignment requirement. If implementation details need to change because of computational constraints, explain the reason and ask me before changing the scientific design.
Part 5: Report and submission

* Write a report of at most five pages.
* Answer all five questions in the assignment.
* Include the required plots and tables.
* Record machine specifications and library versions.
* Include a Git repository with its `.git` directory.
* Provide `README.md`, `requirements.txt`, and `hypothesis.md`.
* Include `results/results.csv` and `results/figures/`.
* Implement `bash run_all.sh` to regenerate the full results and `bash run_all.sh --quick` to regenerate results on a small subset in under 15 minutes on a laptop.
* Include the required AI prompts, responses and file classifications in `ai log/`.
* Do not include datasets or saved index files in the final ZIP.
* Do not use the ann-benchmarks framework or copy benchmark numbers from other sources.

3. First inspect the workspace
Before writing code:

1. List the existing files and folders.
2. Locate and read the assignment PDF.
3. Identify whether the SIFT and GloVe datasets are already available.
4. Identify their file formats, paths and sizes without loading all vectors into RAM unnecessarily.
5. Inspect existing Python environments, dependency files, source code and Git status.
6. Check available disk space and system memory when possible.
7. Determine whether FAISS and the other required libraries are installed.
8. Identify any missing files or information that requires my action.

Do not download large datasets, overwrite existing files, delete anything, or make expensive benchmark runs without asking me first.
If the dataset format cannot be determined confidently, inspect the available documentation or metadata before implementing the loader.
4. Create a detailed implementation journal
Create a root-level file named `IMPLEMENTATION_LOG.md`.
This file is mandatory and must be maintained from the first milestone until final submission. It will serve as the technical record used to prepare the final report and explain the code during evaluation.
Write the journal so that another student or evaluator can understand what was implemented, why each decision was made, how it was tested, and which conclusions are supported by measurements.
Required sections

1. Assignment summary and requirements checklist.
2. Environment and machine specifications.
3. Dataset locations, formats, dimensions and validation results.
4. Overall architecture and source-file responsibilities.
5. Milestone-by-milestone implementation history.
6. Important algorithms, formulas and implementation decisions.
7. Experimental hypotheses and their Git commit identifiers.
8. Configuration parameters and the reason for choosing them.
9. Commands executed and their purposes.
10. Unit tests, integration tests and validation results.
11. Bugs encountered, root causes and fixes.
12. Benchmark measurements and links to result files.
13. Figures and tables generated, including what each shows.
14. Limitations, assumptions and unresolved issues.
15. Decisions or actions required from me.
16. A final report preparation section mapping assignment questions to the evidence needed to answer them.

For each milestone, append a dated entry containing:

* Milestone name and status.
* Files created or modified.
* What was implemented.
* Key technical decisions and reasoning.
* Commands executed.
* Tests performed and their actual outcomes.
* Measurements obtained, including units and relevant configurations.
* Any failures, warnings or limitations.
* What the results establish and what they do not establish.
* Remaining work and the next milestone.
* Any manual action required from me.

Use explicit statuses such as `PLANNED`, `IN PROGRESS`, `BLOCKED`, and `COMPLETED`.
Never invent test outcomes, measurements, hardware specifications, or successful execution claims. Clearly distinguish measured results from expected behaviour and proposed plans.
As implementation progresses, update the journal after every meaningful milestone. Do not postpone documentation until the end.
5. Maintain a requirements and milestone checklist
Create or update `README.md` with:

* Assignment overview.
* Current project status.
* Planned directory structure.
* Environment setup instructions.
* Dataset setup instructions.
* Commands implemented so far.
* How to run the available tests.
* Known limitations.
* The next planned milestone.

Use a checklist in both `README.md` and `IMPLEMENTATION_LOG.md` to track the requirements from Parts 1–5.
Mark a requirement complete only when the corresponding code and validation evidence exist.
Create a `requirements.txt` based on actual dependencies. Do not include arbitrary or untested packages.
6. Implement Milestone 1 only
For this first task, implement environment and data validation, without beginning the expensive benchmarks.
Create an appropriate modular project structure, for example:

* `src/data_loader.py`
* `src/environment_info.py`
* `tests/`
* `results/`
* `experiments/`
* `ai log/`

Adapt the layout if the existing repository already has a suitable structure.
Implement the following:
A. Environment inspection

* Record Python version, operating system, CPU details and available RAM where accessible.
* Record installed versions of NumPy and FAISS, if installed.
* Record relevant dependency versions and available execution constraints.
* Save the collected information in a machine-readable file under `results/`.

B. Dataset inspection

* Detect available dataset files and identify their formats.
* Record file sizes, vector counts, dimensionalities, query counts and ground-truth structure where these can be determined.
* Avoid loading the entire dataset into RAM merely to inspect its metadata.
* If the datasets are missing, document the expected paths and the official download source or instructions. Do not download them without my permission.

C. Data validation
Where datasets are available:

* Validate expected shapes and data types.
* Check for NaN and infinite values using a memory-conscious approach.
* Check the query and ground-truth dimensions.
* Validate the 5,000/5,000 query split.
* Implement the required GloVe normalization procedure in the loader or an appropriate helper, without modifying the original dataset files.
* Document zero-norm vectors or other anomalies rather than silently hiding them.
* Add basic tests for dataset metadata, query splitting and normalization.

Do not implement recall evaluation or ANN benchmarks yet. Those belong to later milestones.
D. Reproducibility foundation

* Ensure file paths are configurable and do not depend on my machine's absolute paths.
* Use deterministic behaviour wherever applicable.
* Keep generated metadata and logs separate from source code.
* Do not commit datasets, generated index files or other large artifacts.
* Preserve any existing user changes and Git history.

7. My responsibilities versus your responsibilities
I am learning the implementation, not simply collecting a finished submission. I need to understand the reasoning behind the code for my individual evaluation.
You must do

* Inspect the project and implement Milestone 1.
* Explain the purpose of every important file and module.
* Document all implementation details and actual test results in `IMPLEMENTATION_LOG.md`.
* Keep `README.md` and the requirements checklist current.
* Use small tests and avoid expensive computation during this milestone.
* Identify missing datasets, environment problems and decisions requiring my input.
* Clearly distinguish implemented, tested, planned and unverified functionality.

I must do

* Provide or download the datasets if they are not already available, following the documented instructions.
* Review the architecture and implementation journal.
* Run commands that require my approval or access to resources unavailable to you.
* Read the explanations and ask questions about algorithms or code I do not understand.
* Review and approve each milestone before implementation proceeds.
* Personally review experimental hypotheses and ensure their Git commits precede the associated experiments.
* Verify the final report, submission files and AI-use disclosure before submitting.

Do not assume that I understand an implementation merely because the code runs. Explain non-obvious decisions in straightforward technical language.
8. AI usage documentation
I am using Claude Code as an AI coding assistant.
Create `ai log/ai statement.md` and establish a workflow for preserving the prompts and responses used during implementation.
Record the current implementation prompt and the actual Codex response, where available. Continue recording relevant prompts and responses for subsequent milestones.
Classify every source file as one of:

* `human-written`
* `AI-assisted`
* `AI-generated`

Use accurate classifications based on actual contributions. Do not claim that an AI log is complete if some interactions have not been captured. If Codex cannot automatically export its conversation history, explain the limitation and tell me exactly what I need to save manually.
9. Working rules

* Do not use external benchmark results as substitutes for our measurements.
* Do not hardcode expected experimental outcomes.
* Do not fabricate measurements or test results.
* Do not silently skip failed configurations.
* Do not change experimental settings without documenting the reason.
* Do not mark a milestone complete if its validation tests have not passed.
* Keep the implementation as simple as possible while meeting the assignment requirements.
* Prefer reproducible scripts and structured CSV output over manual measurements.
* Preserve the distinction between tuning queries and final evaluation queries.
* Ask before making significant design changes or launching computationally expensive jobs.

10. What to return when Milestone 1 is finished
Stop after completing Milestone 1 and provide:

1. A summary of the workspace you found.
2. The directory structure created or modified.
3. A list of every file created or modified.
4. A concise explanation of each implementation decision.
5. The exact commands executed.
6. The actual test results, including failures.
7. Environment and dataset findings.
8. Any missing information or manual actions required from me.
9. The current contents or a detailed summary of `IMPLEMENTATION_LOG.md`.
10. The requirements checklist, identifying what is completed, incomplete or blocked.
11. A clear explanation of what Milestone 2 will implement.

Do not begin Milestone 2 until I have reviewed the results and explicitly instructed you to continue.
