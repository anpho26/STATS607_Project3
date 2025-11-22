#!/usr/bin/env python
"""
Empirical computational complexity analysis for Part C (Proposition 2.6).

Runs `src_cli.partc_log_prop26` for a range of M, measures wall-clock
time, fits a log–log slope, saves a CSV and a log–log plot, and prints a
Markdown table for the baseline report.

Usage (from repo root):

    python scripts/complexity_partC.py
    # or with custom M values:
    python scripts/complexity_partC.py --M 100 200 400 800 1600
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


def run_partc_once(
    M: int,
    alpha: float,
    base: str,
    t_vals: List[float],
    n_vals: List[int],
    seed: int,
) -> float:
    """Run src_cli.partc_log_prop26 once and return wall-clock time (seconds)."""
    cmd = (
        [sys.executable, "-m", "src_cli.partc_log_prop26"]
        + ["--alpha", str(alpha)]
        + ["--base", base]
        + ["--seed", str(seed)]
        + ["--M", str(M)]
        + ["--t", *[str(t) for t in t_vals]]
        + ["--n", *[str(n) for n in n_vals]]
    )

    t0 = time.perf_counter()
    # We only care about timing; suppress output
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t1 = time.perf_counter()
    return t1 - t0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Empirical complexity analysis for Part C (M vs runtime)."
    )
    parser.add_argument(
        "--M",
        nargs="+",
        type=int,
        default=[100, 200, 400, 800, 1600],
        help="List of M values (number of Monte Carlo replicates) to test "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--n",
        nargs="+",
        type=int,
        default=[100, 500, 1000],
        help="List of n values used inside Prop 2.6 (default: %(default)s).",
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
        default=2,
        help="Number of repetitions per M to average runtime (default: %(default)s).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/summary",
        help="Directory to write CSV and figure (default: %(default)s).",
    )
    args = parser.parse_args()

    M_values = sorted(args.M)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("[complexity C] Running Part C for M values:", M_values)
    runtimes = []

    for M in M_values:
        print(f"  M={M}: ", end="", flush=True)
        reps_times = []
        for r in range(args.reps):
            dt = run_partc_once(
                M=M,
                alpha=args.alpha,
                base=args.base,
                t_vals=args.t,
                n_vals=args.n,
                seed=args.seed + r,
            )
            reps_times.append(dt)
            print(f"{dt:.3f}s ", end="", flush=True)
        avg = float(np.mean(reps_times))
        runtimes.append(avg)
        print(f"-> avg {avg:.3f}s")

    M_arr = np.array(M_values, dtype=float)
    t_arr = np.array(runtimes, dtype=float)

    # Fit log–log slope
    logM = np.log(M_arr)
    logt = np.log(t_arr)
    slope, intercept = np.polyfit(logM, logt, 1)

    # Save CSV
    csv_path = outdir / "complexity_partC.csv"
    with csv_path.open("w") as f:
        f.write("M,runtime_sec\n")
        for M, t in zip(M_arr, t_arr):
            f.write(f"{int(M)},{t:.6f}\n")
    print(f"[complexity C] Wrote CSV to {csv_path}")

    # Make log–log plot
    fig_path = outdir / "complexity_partC_loglog.png"
    plt.figure()
    plt.loglog(M_arr, t_arr, "o-")
    plt.xlabel("M (log scale)")
    plt.ylabel("runtime (seconds, log scale)")
    plt.title(
        f"Part C runtime vs M (log–log)\n"
        f"n={args.n}, alpha={args.alpha}, base={args.base}, slope≈{slope:.2f}"
    )
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[complexity C] Wrote log–log plot to {fig_path}")

    # Print Markdown table + slope for copy–paste
    print("\nMarkdown table (copy into BASELINE.md):\n")
    print("| M | Runtime (s) |")
    print("|---|-------------|")
    for M, t in zip(M_arr, t_arr):
        print(f"| {int(M)} | {t:.3f} |")

    print("\nEmpirical log–log fit:")
    print(f"- slope ≈ **{slope:.2f}**")
    print(
        "  (runtime ≈ C * M^{slope}; slope near 1 indicates approximately O(M) complexity)"
    )


if __name__ == "__main__":
    main()