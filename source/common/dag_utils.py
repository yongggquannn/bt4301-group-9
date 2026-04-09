"""Shared utilities for Airflow DAG task functions."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_python_script(script_path: Path, cwd: Path | None = None) -> None:
    """Run one project script with the current interpreter and fail on non-zero exit."""
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    subprocess.run(
        [sys.executable, "-B", str(script_path)],
        cwd=str(cwd) if cwd else None,
        check=True,
    )
