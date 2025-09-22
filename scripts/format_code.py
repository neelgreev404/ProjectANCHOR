#!/usr/bin/env python
"""format_code.py – One-shot PEP-8/Black/Ruff formatter.

Execute:
    python format_code.py [--path DIR] [--line-length 88]

Installs Black and Ruff if missing, runs them across the directory tree,
and prints a summary of files reformatted or issues fixed.

Safe to rerun; Black/Ruff are idempotent.
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import List


def ensure_package_installed(pkg: str, version: str | None = None) -> None:
    if importlib.util.find_spec(pkg) is not None:
        return

    spec = f"{pkg}=={version}" if version else pkg
    print(f"Installing {spec}...", flush=True)
    cmd: List[str] = [sys.executable, "-m", "pip", "install", "--quiet", spec]
    try:
        subprocess.run(cmd, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {spec}: {e}")
        sys.exit(1)


def run_command(cmd: List[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        sys.exit(e.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Black + Ruff autofix over tree")
    parser.add_argument(
        "--path", default=".", help="Root directory to format (default: cwd)"
    )
    parser.add_argument(
        "--line-length", type=int, default=88, help="Black line length (default: 88)"
    )
    args = parser.parse_args()

    root = Path(args.path).resolve()
    if not root.exists():
        print(f"Path does not exist: {root}")
        sys.exit(1)

    ensure_package_installed("black", "24.3.0")
    ensure_package_installed("ruff", "0.4.4")

    run_command(
        [sys.executable, "-m", "black", str(root), f"--line-length={args.line_length}"]
    )

    run_command([sys.executable, "-m", "ruff", "check", str(root), "--fix"])

    print("Formatting completed successfully!")


if __name__ == "__main__":
    main()
