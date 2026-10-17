"""Every source file must at least compile: catches syntax errors in scripts no test imports."""

import py_compile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FILES = sorted(p for d in ("src", "experiments") for p in (ROOT / d).rglob("*.py"))


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_compiles(path):
    py_compile.compile(str(path), doraise=True)
