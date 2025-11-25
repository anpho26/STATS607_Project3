#!/usr/bin/env python
"""
Empirical computational complexity analysis for Part A (panels).

Runs `src_cli.parta_panels` for a range of n, measures wall-clock time,
fits a log–log slope, saves a CSV and a log–log plot, and prints a
Markdown table for the baseline report.

Usage (from repo root):

    python scripts/complexity_partA.py
    # or with custom n values:
    python scripts/complexity_partA.py --n 0 100 500 1000 1500
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


def run_parta_once(
    n: int,
    alpha_list: List[float],
    base: str,
    t_vals: List[float],
    M: int,
    N: int,
    seed: int,
    backend: str,
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
        + ["--alpha", *[str(a) for a in alpha_list]]
        + ["--t", *[str(t) for t in t_vals]]
    )

    t0 = time.perf_counter()
    # We only care about timing; suppress output
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t1 = time.perf_counter()
    return t1 - t0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Empirical complexity analysis for Part A (n vs runtime)."
    )
    parser.add_argument(
        "--n",
        nargs="+",
        type=int,
        default=[0, 100, 500, 1000, 1500],
        help="List of n values to test (default: %(default)s).",
    )
    parser.add_argument(
        "--alpha",
        nargs="+",
        type=float,
        default=[1.0, 5.0, 20.0],
        help="List of alpha values used in panels (default: %(default)s).",
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
        "--M",
        type=int,
        default=4000,
        help="Number of Monte Carlo paths M (default: %(default)s).",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=2000,
        help="Grid size N for predictive CDF (default: %(default)s).",
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
        help="Number of repetitions per n to average runtime (default: %(default)s).",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["baseline", "fast"],
        default=["baseline"],
        help="Which backends to benchmark (default: baseline only).",
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

    backends = list(dict.fromkeys(args.backends))

    print("[complexity A] Running Part A for n values:", n_values)
    runtimes_by_backend: dict[str, list[float]] = {be: [] for be in backends}

    for backend in backends:
        print(f"\n[complexity A] Backend={backend}")
        for n in n_values:
            print(f"  n={n}: ", end="", flush=True)
            reps_times = []
            for r in range(args.reps):
                dt = run_parta_once(
                    n=n,
                    alpha_list=args.alpha,
                    base=args.base,
                    t_vals=args.t,
                    M=args.M,
                    N=args.N,
                    seed=args.seed + r,
                    backend=backend,
                )
                reps_times.append(dt)
                print(f"{dt:.3f}s ", end="", flush=True)
            avg = float(np.mean(reps_times))
            runtimes_by_backend[backend].append(avg)
            print(f"-> avg {avg:.3f}s")

    n_arr = np.array(n_values, dtype=float)

    slopes: dict[str, float] = {}
    mask = n_arr > 0
    for backend in backends:
        t_arr = np.array(runtimes_by_backend[backend], dtype=float)
        logn = np.log(n_arr[mask])
        logt = np.log(t_arr[mask])
        slope, intercept = np.polyfit(logn, logt, 1)
        slopes[backend] = slope

    # Save CSV
    csv_path = outdir / "complexity_partA.csv"
    with csv_path.open("w") as f:
        f.write("backend,n,runtime_sec\n")
        for backend in backends:
            t_arr = np.array(runtimes_by_backend[backend], dtype=float)
            for n, t in zip(n_arr, t_arr):
                f.write(f"{backend},{int(n)},{t:.6f}\n")
    print(f"[complexity A] Wrote CSV to {csv_path}")

    # Make log–log plot (skip n=0 in the plot)
    fig_path = outdir / "complexity_partA_loglog.png"
    plt.figure()
    for backend in backends:
        t_arr = np.array(runtimes_by_backend[backend], dtype=float)
        plt.loglog(n_arr[mask], t_arr[mask], "o-", label=f"{backend} (slope≈{slopes[backend]:.2f})")
    plt.xlabel("n (log scale)")
    plt.ylabel("runtime (seconds, log scale)")
    plt.title(
        f"Part A runtime vs n (log–log)\n"
        f"M={args.M}, alpha={args.alpha}, base={args.base}"
    )
    plt.grid(True, which="both", ls=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"[complexity A] Wrote log–log plot to {fig_path}")

    # Print Markdown table + slope for copy–paste
    print("\nMarkdown table (copy into BASELINE.md):\n")
    print("| backend | n | Runtime (s) |")
    print("|---------|---|-------------|")
    for backend in backends:
        t_arr = np.array(runtimes_by_backend[backend], dtype=float)
        for n, t in zip(n_arr, t_arr):
            print(f"| {backend} | {int(n)} | {t:.3f} |")

    print("\nEmpirical log–log fit (excluding n=0):")
    for backend in backends:
        slope = slopes[backend]
        print(f"- backend `{backend}`: slope ≈ **{slope:.2f}**")
    print("  (runtime ≈ C * n^{slope}; slope near 1 indicates approximately O(n) complexity)")


if __name__ == "__main__":
    main()