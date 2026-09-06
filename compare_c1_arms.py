#!/usr/bin/env python
"""Compare the C1 A/B arms and enforce the stock-vs-telemetry null gate.

The telemetry calculator returns the shipped calculator's output verbatim, so
its rows must be identical to stock on every decision column. If they are not,
the wiring is broken and the floored arm's numbers mean nothing -- which is why
that check runs first and reports a non-zero exit status on failure.

Usage:
    python compare_c1_arms.py [--prefix c1]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

LOGDIR = Path(
    os.environ.get("MONTY_LOGS", Path.home() / "tbp/results/monty")
) / "projects/evidence_eval_runs/logs"
TELDIR = Path(
    os.environ.get("MONTY_LOGS", Path.home() / "tbp/results/monty")
) / "projects/evidence_eval_runs/c1_telemetry"

ARMS = ["stock", "telemetry", "floored"]
LM_COUNTS = [1, 5]

# Wall-clock varies run to run and says nothing about the decisions Monty made.
TIMING_COLUMNS = {"time"}

# Order-independent identity for a row, so parallel completion order cannot
# masquerade as a behavioural difference.
SORT_KEYS = [
    "primary_target_object",
    "primary_target_rotation_euler",
    "episode_seed",
    "lm_id",
]


def load_arm(prefix: str, arm: str, lms: int) -> pd.DataFrame | None:
    path = LOGDIR / f"{prefix}_{arm}_{lms}lm_20obj" / "eval_stats.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def canonical(df: pd.DataFrame) -> pd.DataFrame:
    keys = [k for k in SORT_KEYS if k in df.columns]
    out = df.sort_values(keys).reset_index(drop=True) if keys else df
    return out.drop(columns=[c for c in TIMING_COLUMNS if c in out.columns])


def accuracy(df: pd.DataFrame) -> float:
    hit = df["primary_performance"].isin(["correct", "correct_mlh"])
    return 100.0 * hit.mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="c1", help="run-name prefix (use 'smoke_c1' for smoke runs)")
    args = ap.parse_args()

    failures = 0

    print("=" * 72)
    print("PER-ARM SUMMARY")
    print("=" * 72)
    print(f"{'arm':10s} {'LMs':>4s} {'rows':>6s} {'% correct':>10s} {'mean steps':>11s} {'mean rot err':>13s}")
    frames: dict[tuple[str, int], pd.DataFrame] = {}
    for lms in LM_COUNTS:
        for arm in ARMS:
            df = load_arm(args.prefix, arm, lms)
            if df is None:
                print(f"{arm:10s} {lms:4d} {'--':>6s} {'not run':>10s}")
                continue
            frames[(arm, lms)] = df
            rot = df["rotation_error"].dropna()
            print(
                f"{arm:10s} {lms:4d} {len(df):6d} {accuracy(df):9.2f}% "
                f"{df['monty_matching_steps'].mean():11.1f} {rot.mean():13.2f}"
            )

    print()
    print("=" * 72)
    print("NULL GATE -- telemetry must be bit-for-bit identical to stock")
    print("=" * 72)
    for lms in LM_COUNTS:
        stock, tel = frames.get(("stock", lms)), frames.get(("telemetry", lms))
        if stock is None or tel is None:
            print(f"{lms} LM: SKIPPED (one or both arms missing)")
            continue
        a, b = canonical(stock), canonical(tel)
        if a.shape != b.shape:
            print(f"{lms} LM: FAIL -- shape {a.shape} vs {b.shape}")
            failures += 1
            continue
        mismatched = [c for c in a.columns if not a[c].equals(b[c])]
        if mismatched:
            print(f"{lms} LM: FAIL -- {len(mismatched)} column(s) differ: {mismatched[:8]}")
            failures += 1
        else:
            print(f"{lms} LM: PASS -- all {len(a.columns)} decision columns identical over {len(a)} rows")

    print()
    print("=" * 72)
    print("BEHAVIOURAL COMPARISON -- floored vs stock (this one may differ)")
    print("=" * 72)
    for lms in LM_COUNTS:
        stock, flo = frames.get(("stock", lms)), frames.get(("floored", lms))
        if stock is None or flo is None:
            print(f"{lms} LM: SKIPPED (one or both arms missing)")
            continue
        delta = accuracy(flo) - accuracy(stock)
        step_delta = flo["monty_matching_steps"].mean() - stock["monty_matching_steps"].mean()
        print(
            f"{lms} LM: accuracy {accuracy(stock):.2f}% -> {accuracy(flo):.2f}% "
            f"({delta:+.2f} pts), matching steps {step_delta:+.1f}"
        )

    print()
    print("=" * 72)
    print("TELEMETRY SIDE-CHANNEL")
    print("=" * 72)
    for lms in LM_COUNTS:
        d = TELDIR / f"{args.prefix}_telemetry_{lms}lm_20obj"
        if not d.exists():
            print(f"{lms} LM: no telemetry directory at {d}")
            continue
        files = list(d.glob("c1_telemetry_*.npy"))
        size = sum(f.stat().st_size for f in files) / 1e6
        print(f"{lms} LM: {len(files)} worker file(s), {size:.1f} MB total -> {d}")
        if not files:
            print("      WARNING: directory is empty. Was MONTY_C1_TELEMETRY_DIR exported?")

    if failures:
        print(f"\nNULL GATE FAILED in {failures} case(s). Do not trust the floored arm.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
