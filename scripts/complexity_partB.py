#!/usr/bin/env python
"""
Empirical computational complexity analysis for Part B.

Runs `src_cli.partb_log_convergence` for a range of n, measures wall-clock
time, fits a log–log slope, saves a CSV and a log–log plot, and prints a
Markdown table for the baseline report.

Usage (from repo root):

    python scripts/complexity_partB.py
    # or with custom n values:
    python scripts/complexity_partB.py --n 200 500 1000 2000 5000 10000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt


def run_partb_once(
    n: int,
    alpha: float,
    base: str,
    t_vals: List[float],
    seed: int,
) -> float:
    """Run src_cli.partb_log_convergence once and return wall-clock time (seconds)."""
    cmd = (
        [sys.executable, "-m", "src_cli.partb_log_convergence"]
        + ["--n", str(n)]
        + ["--alpha", str(alpha)]
        + ["--base", base]
        + ["--seed", str(seed)]
        + ["--t", *[str(t) for t in t_vals]]
    )

    t0 = time.perf_counter()
    # discard stdout; we only care about timing
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t1 = time.perf_counter()
    return t1 - t0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Empirical complexity analysis for Part B (n vs runtime)."
    )
    parser.add_argument(
        "--n",
        nargs="+",
        type=int,
        default=[200, 500, 1000, 2000, 5000, 10000],
        help="List of n values to test (default: %(default)s).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=5.0,
        help="Dirichlet process concentration parameter alpha (default: %(default)s).",
    )
    parser.add_argument(
        "--base",
        type=str,
        default="uniform",
        help="Base distribution identifier (default: %(default)s).",
    )
    parser.add_argument(
        "--t",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75],
        help="List of t values (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2025,
        help="Random seed (default: %(default)s).",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Number of repetitions per n to average runtime (default: %(default)s).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/summary",
        help="Directory to write CSV and figure (default: %(default)s).",
    )
    args = parser.parse_args()

    n_values = sorted(args.n)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[complexity] Running Part B for n values:", n_values)
    runtimes = []

    for n in n_values:
        print(f"  n={n}: ", end="", flush=True)
        reps_times = []
        for r in range(args.reps):
            dt = run_partb_once(
                n=n,
                alpha=args.alpha,
                base=args.base,
                t_vals=args.t,
                seed=args.seed + r,  # vary seed slightly
            )
            reps_times.append(dt)
            print(f"{dt:.3f}s ", end="", flush=True)
        avg = float(np.mean(reps_times))
        runtimes.append(avg)
        print(f"-> avg {avg:.3f}s")

    n_arr = np.array(n_values, dtype=float)
    t_arr = np.array(runtimes, dtype=float)

    # Fit log–log slope
    logn = np.log(n_arr)
    logt = np.log(t_arr)
    slope, intercept = np.polyfit(logn, logt, 1)

    # Save CSV
    csv_path = outdir / "complexity_partB.csv"
    with csv_path.open("w") as f:
        f.write("n,runtime_sec\n")
        for n, t in zip(n_arr, t_arr):
            f.write(f"{int(n)},{t:.6f}\n")
    print(f"[complexity] Wrote CSV to {csv_path}")

    # Make log–log plot
    fig_path = outdir / "complexity_partB_loglog.png"
    plt.figure()
    plt.loglog(n_arr, t_arr, "o-")
    plt.xlabel("n (log scale)")
    plt.ylabel("runtime (seconds, log scale)")
    plt.title(
        f"Part B runtime vs n (log–log)\n"
        f"alpha={args.alpha}, base={args.base}, slope≈{slope:.2f}"
    )
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[complexity] Wrote log–log plot to {fig_path}")

    # Print Markdown table + slope for copy–paste
    print("\nMarkdown table (copy into BASELINE.md):\n")
    print("| n | Runtime (s) |")
    print("|---|-------------|")
    for n, t in zip(n_arr, t_arr):
        print(f"| {int(n)} | {t:.3f} |")

    print("\nEmpirical log–log fit:")
    print(f"- slope ≈ **{slope:.2f}**")
    print(
        "  (runtime ≈ C * n^{slope}; slope near 1 indicates approximately O(n) complexity)"
    )


if __name__ == "__main__":
    main()