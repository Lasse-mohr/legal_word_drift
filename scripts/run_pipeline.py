"""Run the core pipeline: fetch → preprocess → train → align.

Executes steps 01 through 06 sequentially, stopping on failure.
Metrics (07+) and visualizations (08+) are excluded.

Usage:
    python scripts/run_pipeline.py [--start 1990] [--end 2025]
    python scripts/run_pipeline.py --skip-phrases   # skip step 04
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import os

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(name: str, script: str, args: list[str]) -> None:
    """Run a pipeline step as a subprocess, abort on failure."""
    path = os.path.join(SCRIPTS_DIR, script)
    cmd = [sys.executable, path] + args
    print(f"\n{'='*60}")
    print(f"  Step: {name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\nFATAL: {name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Run the full data pipeline (steps 01-06)",
    )
    parser.add_argument("--start", type=int, default=1990, help="Start year")
    parser.add_argument("--end", type=int, default=2025, help="End year")
    parser.add_argument(
        "--skip-phrases", action="store_true",
        help="Skip step 04 (phrase detection)",
    )
    args = parser.parse_args()

    year_args = ["--start", str(args.start), "--end", str(args.end)]

    run_step("Fetch metadata", "data_processing/01_fetch_metadata.py", year_args)
    run_step("Fetch texts", "data_processing/02_fetch_texts.py", [])
    run_step("Preprocess", "data_processing/03_preprocess.py", year_args)

    if not args.skip_phrases:
        run_step("Detect phrases", "data_processing/04_detect_phrases.py", year_args)

    run_step("Train embeddings", "model/05_train_embeddings.py", year_args)
    run_step("Align embeddings", "model/06_align_embeddings.py", [])

    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
