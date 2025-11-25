#!/usr/bin/env python
"""
Empirical computational complexity analysis for Part C (Proposition 2.6).

Runs `src_cli.partc_log_prop26` for a range of M values, measures wall-clock
time, and fits log–log slopes for one or more backends (baseline vs optimized).
Results are written to a CSV, a log–log plot, and printed as a Markdown table.

Usage (from repo root):

    # default: compare baseline and fast, n_jobs=1
    python scripts/complexity_partC.py

    # custom M values and a single backend
    python scripts/complexity_partC.py --M 100 200 400 800 --backends baseline

    # compare baseline vs fast with custom n_jobs
    python scripts/complexity_partC.py --n-jobs 1 --backends baseline fast
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt


def run_partc_once(
    M: int,
    alpha: float,
    base: str,
    t_vals: List[float],
    n_vals: List[int],
    seed: int,
    backend: str,
    n_jobs: int,
) -> float:
    """Run src_cli.partc_log_prop26 once and return wall-clock time (seconds)."""
    cmd = (
        [sys.executable, "-m", "src_cli.partc_log_prop26"]
        + ["--alpha", str(alpha)]
        + ["--base", base]
        + ["--seed", str(seed)]
        + ["--M", str(M)]
        + ["--backend", backend]
        + ["--n-jobs", str(n_jobs)]
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
        description="Empirical complexity analysis for Part C (M vs runtime) for one or more backends."
    )
    parser.add_argument(
        "--M",
        nargs="+",
        type=int,
        default=[100, 200, 400, 800, 1600],
        help=(
            "List of M values (number of Monte Carlo replicates) to test "
            "(default: %(default)s)."
        ),
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
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["baseline", "fast"],
        default=["baseline", "fast"],
        help=(
            "Backends to compare for Part C complexity (default: %(default)s). "
            "Typically 'baseline' and 'fast'."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Number of parallel workers to use when timing Part C (default: %(default)s).",
    )
    parser.add_argument(
        "--mode",
        choices=["complexity", "speedup"],
        default="complexity",
        help="Mode 'complexity' varies M (current behavior); mode 'speedup' fixes M and varies n_jobs.",
    )
    parser.add_argument(
        "--n-jobs-list",
        nargs="+",
        type=int,
        default=[1, 2, 4],
        help="For mode='speedup': list of n_jobs values to test (default: %(default)s).",
    )
    args = parser.parse_args()

    if args.mode == "complexity":
        M_values = sorted(args.M)
        print("[complexity C] Running Part C for M values:", M_values)
        print(f"[complexity C] Backends: {args.backends}, n_jobs={args.n_jobs}")

        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        backend_results: Dict[str, List[float]] = {}

        for backend in args.backends:
            print(f"\n[complexity C] Backend={backend}")
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
                        backend=backend,
                        n_jobs=args.n_jobs,
                    )
                    reps_times.append(dt)
                    print(f"{dt:.3f}s ", end="", flush=True)
                avg = float(np.mean(reps_times))
                runtimes.append(avg)
                print(f"-> avg {avg:.3f}s")

            backend_results[backend] = runtimes

        M_arr = np.array(M_values, dtype=float)

        # Save CSV with all backends
        csv_path = outdir / "complexity_partC.csv"
        with csv_path.open("w") as f:
            f.write("backend,M,runtime_sec\n")
            for backend, runtimes in backend_results.items():
                for M, t in zip(M_arr, runtimes):
                    f.write(f"{backend},{int(M)},{t:.6f}\n")
        print(f"[complexity C] Wrote CSV to {csv_path}")

        # Make log–log plot with one curve per backend
        fig_path = outdir / "complexity_partC_loglog.png"
        plt.figure()
        slopes = {}
        for backend, runtimes in backend_results.items():
            t_arr = np.array(runtimes, dtype=float)
            logM = np.log(M_arr)
            logt = np.log(t_arr)
            slope, intercept = np.polyfit(logM, logt, 1)
            slopes[backend] = slope
            plt.loglog(M_arr, t_arr, "o-", label=f"{backend} (slope≈{slope:.2f})")

        plt.xlabel("M (log scale)")
        plt.ylabel("runtime (seconds, log scale)")
        plt.title(
            f"Part C runtime vs M (log–log)\n"
            f"n={args.n}, alpha={args.alpha}, base={args.base}, n_jobs={args.n_jobs}"
        )
        plt.grid(True, which="both", ls=":")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"[complexity C] Wrote log–log plot to {fig_path}")

        # Print Markdown table + slopes for copy–paste
        print("\nMarkdown table (copy into OPTIMIZATION.md or report):\n")
        print("| backend | M | Runtime (s) |")
        print("|---------|---|-------------|")
        for backend, runtimes in backend_results.items():
            for M, t in zip(M_arr, runtimes):
                print(f"| {backend} | {int(M)} | {t:.3f} |")

        print("\nEmpirical log–log slopes:")
        for backend, slope in slopes.items():
            print(
                f"- {backend}: slope ≈ **{slope:.2f}** "
                "(runtime ≈ C * M^{slope}; slope near 1 indicates approximately O(M) complexity)"
            )
    else:
        M_values = sorted(args.M)
        if len(M_values) != 1:
            raise ValueError("For mode='speedup', please provide exactly one M value (e.g., --M 400).")
        M_fixed = M_values[0]

        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        nj_list = args.n_jobs_list
        runtimes = []

        print(f"[speedup C] Running Part C for fixed M={M_fixed} with varying n_jobs: {nj_list}")

        for nj in nj_list:
            dt = run_partc_once(
                M=M_fixed,
                alpha=args.alpha,
                base=args.base,
                t_vals=args.t,
                n_vals=args.n,
                seed=args.seed,
                backend="fast",
                n_jobs=nj,
            )
            runtimes.append(dt)
            print(f"  n_jobs={nj}: {dt:.3f}s")

        T1 = runtimes[0]
        speedups = [T1 / dt for dt in runtimes]

        csv_path = outdir / "speedup_partC.csv"
        with csv_path.open("w") as f:
            f.write("n_jobs,runtime_sec,speedup\n")
            for nj, dt, sp in zip(nj_list, runtimes, speedups):
                f.write(f"{nj},{dt:.6f},{sp:.6f}\n")
        print(f"[speedup C] Wrote CSV to {csv_path}")

        print("\nMarkdown table (copy into OPTIMIZATION.md):\n")
        print("| n_jobs | Runtime (s) | Speedup vs 1 job |")
        print("|--------|-------------|------------------|")
        for nj, dt, sp in zip(nj_list, runtimes, speedups):
            print(f"| {nj} | {dt:.3f} | {sp:.2f}× |")


if __name__ == "__main__":
    main()