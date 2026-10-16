"""Record machine specification and library versions to results/metadata/environment.json.

Usage:  python -m src.environment_info
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata

from src.config import METADATA_DIR

PACKAGES = ("numpy", "faiss-cpu", "h5py", "psutil", "pytest", "matplotlib", "pandas", "hnswlib")


def _cpu_name() -> str:
    """platform.processor() is vague on Windows/Linux, so ask the OS for the marketing name."""
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=30)
            if out.stdout.strip():
                return out.stdout.strip()
        elif sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=30)
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor()


def collect() -> dict:
    info: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_name": _cpu_name(),
        "os_cpu_count_logical": os.cpu_count(),
        "env_OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "packages": {},
    }
    try:
        import psutil
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        info["ram_total_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
        info["ram_available_gb_at_collection"] = round(psutil.virtual_memory().available / 1024**3, 2)
        freq = psutil.cpu_freq()
        info["cpu_freq_max_mhz"] = freq.max if freq else None
    except ImportError:
        info["psutil"] = "not installed; RAM and physical core count unavailable"

    for pkg in PACKAGES:
        try:
            info["packages"][pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            info["packages"][pkg] = None

    try:
        import faiss
        info["faiss_omp_max_threads_default"] = faiss.omp_get_max_threads()
        try:
            info["faiss_compile_options"] = faiss.get_compile_options()
        except AttributeError:
            info["faiss_compile_options"] = None
    except ImportError:
        info["faiss_omp_max_threads_default"] = None
    return info


def main() -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    info = collect()
    out = METADATA_DIR / "environment.json"
    out.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
