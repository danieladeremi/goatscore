#!/usr/bin/env python3
"""
Run the NBA GOAT pipeline outside Zerve by executing script blocks in graph order.

This runner preserves one shared global namespace across steps, which mirrors how
canvas blocks pass variables (career_df, era_adj_career_df, goat_df, etc.).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
import tokenize

def _configure_stdio_utf8() -> None:
    # Several upstream scripts print Unicode glyphs (e.g., checkmarks).
    # Force UTF-8 on Windows terminals to avoid cp1252 encode crashes.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


DEFAULT_STEP_ORDER = [
    # Data build + validation
    "load_historical_players.py",
    "scrape_career_stats.py",
    "build_full_dataset.py",
    "scrape_honors_data.py",
    "clean_merge_dataset.py",
    "validate_and_save.py",
    # Modeling + scoring branches
    "era_adjustment_engine.py",
    "era_adjustment_visualization.py",
    "age_curve_model.py",
    "career_projection_engine.py",
    "projection_visualizations.py",
    "goat_score_engine.py",
    "goat_score_visualizations.py",
    "goat_bias_corrections.py",
    "corrected_goat_visualizations.py",
    "goat_engine_v2.py",
    "goat_score_engine_v3.py",
    "age_curve_trajectory_data.py",
    "goat_engine_v4.py",
    "goat_engine_v4_top25.py",
]

DATA_ONLY_STEPS = [
    "load_historical_players.py",
    "scrape_career_stats.py",
    "build_full_dataset.py",
    "scrape_honors_data.py",
    "clean_merge_dataset.py",
    "validate_and_save.py",
]


def _read_source(path: Path) -> str:
    # tokenize.open honors PEP-263 encoding comments if present.
    with tokenize.open(str(path)) as f:
        return f.read()


def run_steps(script_dir: Path, steps: list[str], stop_on_error: bool) -> int:
    env: dict[str, object] = {
        "__name__": "__main__",
        "__file__": "",
        "__package__": None,
        "__cached__": None,
    }

    os.chdir(script_dir)
    print(f"[runner] working directory: {script_dir}")
    print(f"[runner] total steps: {len(steps)}")

    failures = 0

    for idx, step in enumerate(steps, start=1):
        path = script_dir / step
        if not path.exists():
            print(f"\n[{idx}/{len(steps)}] MISSING: {step}")
            failures += 1
            if stop_on_error:
                return failures
            continue

        print(f"\n[{idx}/{len(steps)}] RUNNING: {step}")
        start = time.time()

        try:
            env["__file__"] = str(path)
            source = _read_source(path)
            code = compile(source, str(path), "exec")
            exec(code, env)
            elapsed = time.time() - start
            print(f"[runner] OK: {step} ({elapsed:.1f}s)")
        except Exception:
            failures += 1
            elapsed = time.time() - start
            print(f"[runner] FAILED: {step} ({elapsed:.1f}s)")
            traceback.print_exc()
            if stop_on_error:
                return failures

    print("\n[runner] completed.")
    print(f"[runner] failures: {failures}")
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the NBA GOAT pipeline locally with shared globals."
    )
    parser.add_argument(
        "--mode",
        choices=["full", "data-only"],
        default="full",
        help="Run all graph blocks or only data-build blocks.",
    )
    parser.add_argument(
        "--steps",
        default="",
        help="Comma-separated script names to override mode order.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining steps after a failure.",
    )
    return parser.parse_args()


def main() -> int:
    _configure_stdio_utf8()
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    if args.steps.strip():
        steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    elif args.mode == "data-only":
        steps = DATA_ONLY_STEPS
    else:
        steps = DEFAULT_STEP_ORDER

    failures = run_steps(
        script_dir=script_dir,
        steps=steps,
        stop_on_error=not args.continue_on_error,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
