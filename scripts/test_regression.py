#!/usr/bin/env python
"""
Regression checks for Unit 3 optimizations (Parts A and C).

Run with:
    python -m scripts.test_regression
or, if you use pytest:
    pytest scripts.test_regression.py
"""

from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from typing import List

import numpy as np
import pandas as pd

from src_cli.partc_log_prop26 import run_one_replicate
import subprocess
import sys


# ==========================
# Helpers
# ==========================

def _summarize(arr: np.ndarray) -> dict:
    """Simple summary stats helper (not used yet, but handy if we extend Part A tests)."""
    return {
        "mean": float(np.mean(arr)),
        "var": float(np.var(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
    }


# ==========================
# Part C: baseline vs fast backend
# ==========================

def _regression_partC_baseline_vs_fast(
    n: int = 200,
    alpha: float = 5.0,
    base: str = "uniform",
    tvals: List[float] | None = None,
    L: int = 2000,
    level: float = 0.95,
    n_reps: int = 100,
    seed: int = 999,
) -> pd.DataFrame:
    """
    Run a small Part C experiment with both backends and return a DataFrame
    summarizing coverage / width across reps.

    Much smaller than the full project run so it finishes quickly but still
    gives a sense of agreement.
    """
    if tvals is None:
        tvals = [0.25, 0.5]

    rows_baseline: list[dict] = []
    rows_fast: list[dict] = []

    for rep in range(n_reps):
        rows_baseline.extend(
            run_one_replicate(
                rep=rep,
                n=n,
                alpha=alpha,
                base=base,
                tvals=tvals,
                L=L,
                level=level,
                seed=seed,
                backend="baseline",
            )
        )
        rows_fast.extend(
            run_one_replicate(
                rep=rep,
                n=n,
                alpha=alpha,
                base=base,
                tvals=tvals,
                L=L,
                level=level,
                seed=seed,
                backend="fast",
            )
        )

    df_base = pd.DataFrame(rows_baseline)
    df_fast = pd.DataFrame(rows_fast)

    g_base = df_base.groupby("t").agg(
        coverage_mean=("covered", "mean"),
        width_mean=("width", "mean"),
    )
    g_fast = df_fast.groupby("t").agg(
        coverage_mean=("covered", "mean"),
        width_mean=("width", "mean"),
    )

    out = pd.concat(
        [g_base.add_suffix("_baseline"), g_fast.add_suffix("_fast")],
        axis=1,
    )
    return out


def test_partC_regression():
    """
    Check that Part C coverage and CI widths are similar for baseline vs fast.

    This is a statistical regression test (not exact equality):
    we require that differences are small.
    """
    summary = _regression_partC_baseline_vs_fast()

    cov_diff = (summary["coverage_mean_baseline"] - summary["coverage_mean_fast"]).abs()
    width_diff = (summary["width_mean_baseline"] - summary["width_mean_fast"]).abs()

    print("\n[Part C regression summary]\n", summary)

    # tolerances are heuristic; adjust if they're too tight
    assert (cov_diff < 0.05).all()
    assert (width_diff < 0.05).all()


def test_partC_fast_backend_sanity():
    """
    Very small sanity check that fast backend produces plausible probabilities.
    """
    rows = run_one_replicate(
        rep=0,
        n=50,
        alpha=5.0,
        base="uniform",
        tvals=[0.25, 0.5],
        L=500,
        level=0.95,
        seed=1234,
        backend="fast",
    )
    for row in rows:
        assert 0.0 <= row["Pn"] <= 1.0
        assert 0.0 <= row["Fhat"] <= 1.0
        assert row["Vnt"] >= 0.0
        assert row["width"] >= 0.0


# ==========================
# Part A: smoke + output check for both backends
# ==========================

def _run_parta_cli(backend: str) -> None:
    """
    Run a *small* Part A configuration via the CLI to ensure both backends run
    and produce a figure. This is a smoke test that the optimized code does not
    change the overall behavior / outputs pipeline.
    """
    # You can tweak these args to match your typical Part A settings; keep them small.
    cmd = [
        sys.executable,
        "-m",
        "src_cli.parta_panels",
        "--base",
        "uniform",
        "--t",
        "0.5",
        "--alpha",
        "5.0",
        "--n",
        "200",
        "--M",
        "500",
        "--N",
        "500",
        "--seed",
        "101",
        "--backend",
        backend,
    ]
    # If your CLI doesn't have --output yet, it will just ignore this; the test
    # will need to look for the default filename instead.
    subprocess.run(cmd, check=True)

def test_parta_smoke_both_backends():
    """
    Smoke test: run a small Part A job for both baseline and fast backends,
    and check that they complete and produce some figure file.

    Notes:
    - This doesn't yet compare *numerical* results, but combined with the
      Part C backend regression, it gives reasonable confidence.
    """
    n = 200
    M = 500
    N = 500
    base = "uniform"

    # Run both backends; they will overwrite or add to the same figures directory.
    _run_parta_cli("baseline")
    _run_parta_cli("fast")

    fig_dir = ROOT / "results" / "figures"
    assert fig_dir.exists(), f"Figure directory {fig_dir} was not created by Part A."

    png_files = list(fig_dir.glob("*.png"))
    assert png_files, f"No PNG figures found in {fig_dir} after running Part A backends."

    # At least one non-empty PNG file indicates the plotting pipeline ran.
    nonempty = [f for f in png_files if f.stat().st_size > 0]
    assert nonempty, "All PNG figures are empty (size 0)."

    print(f"[Part A smoke] Found {len(nonempty)} non-empty PNG figures in {fig_dir}.")

if __name__ == "__main__":
    # Manual runner: execute tests
    test_partC_regression()
    test_partC_fast_backend_sanity()
    test_parta_smoke_both_backends()
    print("\nAll regression checks completed.")