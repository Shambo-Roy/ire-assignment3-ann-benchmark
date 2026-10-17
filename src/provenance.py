"""Git provenance for result rows and the hypothesis guard (decision A6).

require_committed_hypothesis() must be called by the Part 3 / Part 4 drivers before any data
generation, index build or query. It is procedural evidence of ordering, not tamper-proof:
clocks and git dates can be set arbitrarily (see IMPLEMENTATION_LOG.md, A6).
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.config import REPO_ROOT

HYPOTHESIS_FILE = "hypothesis.md"
PLACEHOLDER = "_Not yet written"
MIN_WORDS = 30
GUARD_EXIT_CODE = 3


class HypothesisGuardError(RuntimeError):
    exit_code = GUARD_EXIT_CODE


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=check)


def git_info(repo: Path = REPO_ROOT) -> dict:
    """HEAD commit and whether tracked files differ from it (untracked files are ignored)."""
    try:
        head = _git(repo, "rev-parse", "HEAD").stdout.decode().strip()
        dirty = bool(_git(repo, "status", "--porcelain", "--untracked-files=no").stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "unavailable", "git_dirty": True}
    return {"git_commit": head, "git_dirty": dirty}


def run_metadata(experiment_id: str, repo: Path = REPO_ROOT) -> dict:
    """Pre-run provenance columns shared by every row of one run."""
    meta = {"run_id": uuid.uuid4().hex[:12], "experiment_id": experiment_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    meta.update(git_info(repo))
    meta["reportable"] = not meta["git_dirty"]
    return meta


def _section(text: str, section: str) -> str | None:
    """Body of the '## <section>...' heading up to the next '## ' heading."""
    pat = re.compile(rf"^##\s+{re.escape(section)}\b.*?$(.*?)(?=^##\s|\Z)", re.M | re.S)
    m = pat.search(text)
    return m.group(1).strip() if m else None


def require_committed_hypothesis(section: str, repo: Path = REPO_ROOT, require_clean_tree: bool = True) -> dict:
    """Refuse (HypothesisGuardError) unless the hypothesis section is written and committed.

    section: heading prefix, e.g. "Part 3" or "Part 4".
    Returns hypothesis_commit, hypothesis_sha256 and the pre-run HEAD.
    """
    def fail(msg: str):
        raise HypothesisGuardError(f"hypothesis guard refused to start ({section}): {msg}")

    try:
        _git(repo, "rev-parse", "--is-inside-work-tree")
    except (OSError, subprocess.CalledProcessError):
        fail("not a git repository or git is unavailable")
    if _git(repo, "ls-files", "--error-unmatch", HYPOTHESIS_FILE, check=False).returncode != 0:
        fail(f"{HYPOTHESIS_FILE} is not tracked by git")
    if _git(repo, "status", "--porcelain", "--", HYPOTHESIS_FILE).stdout.strip():
        fail(f"{HYPOTHESIS_FILE} has uncommitted (staged or unstaged) changes")
    committed = _git(repo, "show", f"HEAD:{HYPOTHESIS_FILE}").stdout.replace(b"\r\n", b"\n")
    working = (repo / HYPOTHESIS_FILE).read_bytes().replace(b"\r\n", b"\n")
    if committed != working:
        fail(f"working copy of {HYPOTHESIS_FILE} differs from HEAD")

    try:
        text = committed.decode("utf-8")
    except UnicodeDecodeError:
        fail(f"{HYPOTHESIS_FILE} is not valid UTF-8; re-save it as UTF-8 and commit")
    body = _section(text, section)
    if body is None:
        fail(f"no '## {section}' section in {HYPOTHESIS_FILE}")
    if PLACEHOLDER in body:
        fail("section still contains the template placeholder")
    words = re.findall(r"\w+", body)
    if len(words) < MIN_WORDS:
        fail(f"section has {len(words)} words; at least {MIN_WORDS} required")

    h_commit = _git(repo, "log", "-1", "--format=%H", "--", HYPOTHESIS_FILE).stdout.decode().strip()
    if _git(repo, "merge-base", "--is-ancestor", h_commit, "HEAD", check=False).returncode != 0:
        fail(f"hypothesis commit {h_commit} is not an ancestor of HEAD")
    info = git_info(repo)
    if require_clean_tree and info["git_dirty"]:
        fail("tracked source files have uncommitted changes (use --dev for non-reportable runs)")
    return {"hypothesis_commit": h_commit,
            "hypothesis_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "pre_run_head": info["git_commit"]}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
