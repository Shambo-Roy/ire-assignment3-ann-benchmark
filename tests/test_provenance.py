import subprocess

import pytest

from src.provenance import HypothesisGuardError, git_info, require_committed_hypothesis
from src.results_io import COLUMNS, read_rows, replace_rows

TEMPLATE = """# Hypotheses

## Part 3 — Compression

_Not yet written. To be filled in by the author before the Part 3 milestone._

## Part 4 — Dimensionality

_Not yet written. To be filled in by the author before the Part 4 milestone._
"""
WRITTEN = TEMPLATE.replace(
    "_Not yet written. To be filled in by the author before the Part 3 milestone._",
    " ".join(["I expect fewer PQ sub-vectors to lower recall because each code keeps less information,"]
             + ["and re-ranking the top 100 candidates to recover most of the lost recall at small cost."] * 2))


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "config", "user.email", "t@example.com")
    git(tmp_path, "config", "user.name", "t")
    git(tmp_path, "config", "core.autocrlf", "false")
    (tmp_path / "hypothesis.md").write_text(TEMPLATE, encoding="utf-8", newline="\n")
    (tmp_path / "code.py").write_text("x = 1\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "template")
    return tmp_path


def commit_hypothesis(repo, text=WRITTEN):
    (repo / "hypothesis.md").write_text(text, encoding="utf-8", newline="\n")
    git(repo, "commit", "-q", "-am", "hypothesis")


def test_guard_refuses_template(repo):
    with pytest.raises(HypothesisGuardError, match="placeholder"):
        require_committed_hypothesis("Part 3", repo)


def test_guard_refuses_uncommitted_hypothesis(repo):
    (repo / "hypothesis.md").write_text(WRITTEN, encoding="utf-8", newline="\n")
    with pytest.raises(HypothesisGuardError, match="uncommitted"):
        require_committed_hypothesis("Part 3", repo)


def test_guard_refuses_staged_but_uncommitted(repo):
    (repo / "hypothesis.md").write_text(WRITTEN, encoding="utf-8", newline="\n")
    git(repo, "add", "hypothesis.md")
    with pytest.raises(HypothesisGuardError, match="uncommitted"):
        require_committed_hypothesis("Part 3", repo)


def test_guard_refuses_missing_section_and_short_text(repo):
    commit_hypothesis(repo, "# Hypotheses\n\n## Part 3\n\nToo short.\n")
    with pytest.raises(HypothesisGuardError, match="words"):
        require_committed_hypothesis("Part 3", repo)
    with pytest.raises(HypothesisGuardError, match="no '## Part 4'"):
        require_committed_hypothesis("Part 4", repo)


def test_guard_refuses_non_utf8_file_cleanly(repo):
    (repo / "hypothesis.md").write_bytes(WRITTEN.encode("cp1252"))   # em-dash -> 0x97
    git(repo, "commit", "-q", "-am", "cp1252 hypothesis")
    with pytest.raises(HypothesisGuardError, match="UTF-8"):
        require_committed_hypothesis("Part 3", repo)


def test_guard_refuses_untracked_file(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "hypothesis.md").write_text(WRITTEN, encoding="utf-8")
    with pytest.raises(HypothesisGuardError, match="not tracked"):
        require_committed_hypothesis("Part 3", tmp_path)


def test_guard_refuses_dirty_source_unless_dev(repo):
    commit_hypothesis(repo)
    (repo / "code.py").write_text("x = 2\n")
    with pytest.raises(HypothesisGuardError, match="tracked source"):
        require_committed_hypothesis("Part 3", repo)
    assert require_committed_hypothesis("Part 3", repo, require_clean_tree=False)["hypothesis_commit"]


def test_guard_accepts_committed_hypothesis_and_reports_commit(repo):
    commit_hypothesis(repo)
    h = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    git(repo, "commit", "-q", "--allow-empty", "-m", "later work")
    info = require_committed_hypothesis("Part 3", repo)
    assert info["hypothesis_commit"] == h
    assert info["pre_run_head"] != h
    assert len(info["hypothesis_sha256"]) == 64
    # the Part 4 section is still a template
    with pytest.raises(HypothesisGuardError):
        require_committed_hypothesis("Part 4", repo)


def test_git_info_tracks_dirty_tracked_files_only(repo):
    assert git_info(repo)["git_dirty"] is False
    (repo / "untracked.txt").write_text("u")
    assert git_info(repo)["git_dirty"] is False
    (repo / "code.py").write_text("x = 3\n")
    assert git_info(repo)["git_dirty"] is True


# ------------------------------------------------------------ results.csv

def test_replace_rows_is_idempotent_and_keeps_other_experiments(tmp_path):
    p = tmp_path / "results.csv"
    a = {"experiment_id": "p1", "dataset_id": "sift", "n_base": 10, "qps_median": 1.0, "build_params": {"M": 16}}
    b = {"experiment_id": "p1", "dataset_id": "glove", "n_base": 10, "qps_median": 2.0}
    replace_rows([a], p)
    replace_rows([b], p)
    replace_rows([dict(a, qps_median=3.0)], p)
    rows = read_rows(p)
    assert len(rows) == 2
    assert {r["dataset_id"]: r["qps_median"] for r in rows} == {"sift": "3.0", "glove": "2.0"}
    assert list(rows[0].keys()) == COLUMNS
    assert [r for r in rows if r["dataset_id"] == "sift"][0]["build_params"] == '{"M": 16}'


def test_replace_rows_rejects_unknown_columns(tmp_path):
    with pytest.raises(KeyError):
        replace_rows([{"experiment_id": "x", "dataset_id": "d", "n_base": 1, "bogus": 1}], tmp_path / "r.csv")
