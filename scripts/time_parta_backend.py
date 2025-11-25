#!/usr/bin/env python
"""
Compare baseline vs fast backend for Part A panels.

Runs `python -m src_cli.parta_panels` with backend="baseline" and backend="fast"
for a fixed configuration, measures wall-clock time, and prints a small summary
and a Markdown table for use in the optimization report.

Usage (from repo root):

    python scripts/time_parta_backend.py
    # or with custom settings, e.g.:
    python scripts/time_parta_backend.py --n 1000 --M 4000 --N 2000 --reps 3
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from typing import List


def run_parta(
    backend: str,
    base: str,
    ts: List[float],
    alphas: List[float],
    n: int,
    M: int,
    N: int,
    seed: int,
) -> float:
    """Run src_cli.parta_panels once and return wall-clock time (seconds)."""
    cmd = (
        [sys.executable, "-m", "src_cli.parta_panels"]
        + ["--base", base]
        + ["--n", str(n)]
        + ["--M", str(M)]
        + ["--N", str(N)]
        + ["--seed", str(seed)]
        + ["--backend", backend]
        + ["--t", *[str(t) for t in ts]]
        + ["--alpha", *[str(a) for a in alphas]]
    )

    t0 = time.perf_counter()
    # Let the CLI print its usual "[ok] wrote ..." lines.
    subprocess.run(cmd, check=True)
    t1 = time.perf_counter()
    return t1 - t0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Time baseline vs fast backend for Part A panels."
    )
    ap.add_argument("--base", choices=["uniform", "normal"], default="uniform")
    ap.add_argument(
        "--t",
        dest="ts",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75],
        help="Thresholds t (default: %(default)s).",
    )
    ap.add_argument(
        "--alpha",
        dest="alphas",
        type=float,
        nargs="+",
        default=[1.0, 5.0, 20.0],
        help="Alpha values (default: %(default)s).",
    )
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--M", type=int, default=4000)
    ap.add_argument("--N", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Number of repetitions per backend (default: %(default)s).",
    )
    args = ap.parse_args()

    backends = ["baseline", "fast"]
    results: dict[str, float] = {}

    print("[time Part A] baseline vs fast")
    for backend in backends:
        times: list[float] = []
        print(f"  backend={backend}: ", end="", flush=True)
        for r in range(args.reps):
            dt = run_parta(
                backend=backend,
                base=args.base,
                ts=args.ts,
                alphas=args.alphas,
                n=args.n,
                M=args.M,
                N=args.N,
                seed=args.seed + r,  # vary seed slightly
            )
            times.append(dt)
            print(f"{dt:.3f}s ", end="", flush=True)
        avg = sum(times) / len(times)
        results[backend] = avg
        print(f"-> avg {avg:.3f}s")

    # Markdown table
    print("\nMarkdown table (copy into OPTIMIZATION.md):\n")
    print("| Backend   | Avg runtime (s) |")
    print("|-----------|-----------------|")
    for backend in backends:
        print(f"| {backend} | {results[backend]:.3f} |")

    # Speedup
    baseline = results["baseline"]
    fast = results["fast"]
    if fast > 0:
        speedup = baseline / fast
        print(f"\nSpeedup (baseline / fast): {speedup:.2f}×")
    else:
        print("\nSpeedup (baseline / fast): undefined (fast runtime ≈ 0)")


if __name__ == "__main__":
    main()