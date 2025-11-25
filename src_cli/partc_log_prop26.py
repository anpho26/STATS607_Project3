import argparse
import math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import List
import os

import numpy as np
import pandas as pd


# ---- base CDF and base sampler ----
def G0_cdf(t, base: str = "uniform") -> float:
    """Base CDF G0(t) for the chosen prior base.

    Parameters
    ----------
    t : float
        Threshold.
    base : {"uniform","normal"}
        Name of the base distribution.

    Returns
    -------
    float
        G0(t). Uses exact formulas; Φ(t) via `erf` for Normal.
    """
    if base == "uniform":
        if t <= 0.0:
            return 0.0
        if t >= 1.0:
            return 1.0
        return float(t)
    elif base == "normal":
        # Φ(t) via erf
        return 0.5 * (1.0 + math.erf(t / math.sqrt(2.0)))
    else:
        raise ValueError(f"unknown base: {base}")


def sample_from_base(rng: np.random.Generator, base: str = "uniform") -> float:
    """Draw a single sample from the base G0 using the provided RNG."""
    if base == "uniform":
        return float(rng.random())
    elif base == "normal":
        return float(rng.normal())
    else:
        raise ValueError(f"unknown base: {base}")


# ---- baseline one-step Pólya update (list-based) ----
def draw_polya_next(xs: list[float], alpha: float, rng: np.random.Generator, base: str = "uniform") -> float:
    """One-step Blackwell–MacQueen Pólya update (baseline list-based version).

    With probability α/(α+m) draw a fresh atom from G0; otherwise pick
    uniformly among the existing atoms in `xs`.
    """
    m = len(xs)                 # current size
    p_new = alpha / (alpha + m) # prob of a fresh draw from G0
    if rng.random() < p_new:
        return sample_from_base(rng, base)
    else:
        j = rng.integers(0, m)  # pick an existing atom uniformly
        return float(xs[j])


# ---- fast Pólya prefix + continuation using NumPy arrays ----
def run_one_replicate(
    rep: int,
    n: int,
    alpha: float,
    base: str,
    tvals: List[float],
    L: int,
    level: float,
    seed: int,
    backend: str = "baseline",
) -> list[dict]:
    """Run a single Monte Carlo replicate for Proposition 2.6.

    This computes V_{n,t}, P_n(t), continues the SAME urn by L steps, and
    forms a Wald CI for \u0303F(t). The returned list contains one row per t.

    Parameters
    ----------
    rep : int
        Replicate index (0-based).
    n : int
        Sample size for the prefix.
    alpha : float
        DP concentration parameter.
    base : {"uniform","normal"}
        Base distribution name.
    tvals : list of float
        Thresholds t for which Proposition 2.6 is evaluated.
    L : int
        Tail length for continuation.
    level : float
        Confidence level (e.g. 0.95).
    seed : int
        Global seed; each (rep, n) pair uses a deterministic offset.
    backend : {"baseline","fast"}
        Choice of implementation: "baseline" uses the original list-based
        Pólya update; "fast" uses a NumPy-backed urn to reduce Python overhead.

    Returns
    -------
    list of dict
        Rows to be appended to the output DataFrame, one per t.
    """
    # Independent per-(rep,n) seed to avoid path reuse across settings.
    rng = np.random.default_rng(seed + 7919 * rep + 104729 * n)

    # Initialize bookkeeping for each t: K_m(t), previous P_{m-1}, running sum for V_{n,t}
    Km = {t: 0 for t in tvals}
    P_prev = {t: G0_cdf(t, base) for t in tvals}  # P0(t) = G0(t)
    Vnt = {t: 0.0 for t in tvals}

    backend = backend.lower()
    if backend not in {"baseline", "fast"}:
        raise ValueError(f"unknown backend={backend!r}; expected 'baseline' or 'fast'")

    # ---- generate prefix x1..xn from the Pólya urn ----
    if backend == "baseline":
        xs: list[float] = []
        for m in range(1, n + 1):
            # draw x_m using the same urn
            x_m = draw_polya_next(xs, alpha, rng, base=base)
            xs.append(x_m)

            # update counts and P_m, accumulate m^2 (P_m - P_{m-1})^2
            for t in tvals:
                if x_m <= t:
                    Km[t] += 1
                Pm = (alpha * G0_cdf(t, base) + Km[t]) / (alpha + m)
                Vnt[t] += (m**2) * (Pm - P_prev[t])**2
                P_prev[t] = Pm  # becomes P_m for next step
    else:
        # fast backend: use a NumPy array xs of length n+L, but still perform
        # the Pólya updates sequentially to preserve the same algorithm.
        xs_arr = np.empty(n + L, dtype=float)
        for m in range(1, n + 1):
            m_curr = m - 1  # current size before drawing X_m
            if m_curr == 0:
                x_m = sample_from_base(rng, base)
            else:
                p_new = alpha / (alpha + m_curr)
                if rng.random() < p_new:
                    x_m = sample_from_base(rng, base)
                else:
                    j = rng.integers(0, m_curr)
                    x_m = float(xs_arr[j])
            xs_arr[m_curr] = x_m

            for t in tvals:
                if x_m <= t:
                    Km[t] += 1
                Pm = (alpha * G0_cdf(t, base) + Km[t]) / (alpha + m)
                Vnt[t] += (m**2) * (Pm - P_prev[t])**2
                P_prev[t] = Pm  # becomes P_m for next step

    # finalize V_{n,t}
    for t in tvals:
        Vnt[t] /= n

    # P_n(t) for each t
    Pn = {t: (alpha * G0_cdf(t, base) + Km[t]) / (alpha + n) for t in tvals}

    # ---- continuation: extend the SAME urn by L steps and estimate F~(t) ----
    tail_leq = {t: 0 for t in tvals}

    if backend == "baseline":
        # xs is a Python list; continue the same urn
        for j in range(L):
            x_next = draw_polya_next(xs, alpha, rng, base=base)
            xs.append(x_next)
            for t in tvals:
                if x_next <= t:
                    tail_leq[t] += 1
    else:
        # xs_arr already has prefix in positions 0..n-1; continue in  n..n+L-1
        xs_arr = xs_arr  # for clarity
        for k in range(L):
            m_curr = n + k  # current size before drawing X_{n+k+1}
            p_new = alpha / (alpha + m_curr)
            if rng.random() < p_new:
                x_next = sample_from_base(rng, base)
            else:
                j = rng.integers(0, m_curr)
                x_next = float(xs_arr[j])
            xs_arr[m_curr] = x_next
            for t in tvals:
                if x_next <= t:
                    tail_leq[t] += 1

    Fhat = {t: tail_leq[t] / float(L) for t in tvals}

    # z critical: avoid SciPy; exact for 0.95, otherwise lock to 0.95 in main
    if abs(level - 0.95) < 1e-12:
        z = 1.959963984540054
    else:
        z = 1.959963984540054

    rows: list[dict] = []
    for t in tvals:
        se = math.sqrt(max(Vnt[t], 1e-12) / n)
        lo = Pn[t] - z * se
        hi = Pn[t] + z * se
        covered = int(lo <= Fhat[t] <= hi)
        rows.append(
            {
                "rep": rep,
                "n": n,
                "alpha": alpha,
                "base": base,
                "t": t,
                "Pn": Pn[t],
                "Vnt": Vnt[t],
                "level": level,
                "z": z,
                "L": L,
                "Fhat": Fhat[t],
                "lo": lo,
                "hi": hi,
                "covered": covered,
                "width": 2 * z * se,
            }
        )
    return rows


def _worker_run_one_replicate(args_tuple) -> list[dict]:
    """Wrapper for ProcessPoolExecutor: unpack args tuple and call run_one_replicate."""
    return run_one_replicate(*args_tuple)


def main() -> None:
    """Compute Proposition 2.6 predictive CIs for \u0303F(t) via continuation.

    Outline
    -------
    For each n in --n and each replication:
      1) Generate x_1..x_n from the SAME Pólya urn with base G0 and α.
      2) Track K_m(t) and P_m(t) iteratively for each t; accumulate
         V_{n,t} = (1/n) * Σ_{m=1}^n m^2 (P_m − P_{m−1})^2.
      3) Compute P_n(t) at the end of the prefix.
      4) CONTINUE THE SAME URN by L extra draws to estimate \u0303F(t) as
         F̂(t) = (1/L) * Σ 1{x_{n+ℓ} ≤ t}.
      5) Form Wald CI: P_n(t) ± z * sqrt(V_{n,t}/n), and record coverage of F̂(t).

    Notes
    -----
    - We fix z for 95% CIs without SciPy; other levels warn and still use z≈1.95996.
    - RNG seeding uses a hash of (rep, n) to keep replicates independent.
    - The continuation uses the SAME urn, as required by Prop 2.6.
    - The `backend` argument toggles between the original list-based update
      and a NumPy-backed urn implementation with the same Monte Carlo logic.
    - The `n_jobs` argument enables parallelization across Monte Carlo replicates.
    """
    # Choose defaults based on OPTIMIZED environment variable
    optimized = os.getenv("OPTIMIZED") == "1"
    default_backend = "fast" if optimized else "baseline"
    # If optimized, default to 2; otherwise, stay serial
    default_n_jobs = 2 if optimized else 1

    ap = argparse.ArgumentParser(
        description="Prop 2.6 predictive CIs for F~(t), with target via continuation on the SAME urn."
    )
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--base", choices=["uniform", "normal"], default="uniform")
    ap.add_argument("--t", nargs="+", type=float, required=True, help="thresholds t (one or more)")
    ap.add_argument("--n", nargs="+", type=int, required=True, help="sample sizes n (one or more)")
    ap.add_argument("--M", type=int, default=200, help="number of datasets (MC reps)")
    ap.add_argument("--L", type=int, default=50000, help="tail length for continuation")
    ap.add_argument("--level", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--backend",
        choices=["baseline", "fast"],
        default=default_backend,
        help="Implementation for the Pólya urn: 'baseline' (lists) or 'fast' (NumPy-backed). Default depends on OPTIMIZED.",
    )
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=default_n_jobs,
        help="Number of parallel worker processes for Monte Carlo reps (1 = no parallelism). Default may be >1 if OPTIMIZED=1.",
    )
    args = ap.parse_args()

    # z critical: avoid SciPy; exact for 0.95, warn otherwise
    if abs(args.level - 0.95) < 1e-12:
        z = 1.959963984540054
    else:
        print(f"[warn] level {args.level} not 0.95; using z≈1.95996 anyway.")
        z = 1.959963984540054

    outdir = Path("results/raw")
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    tvals = list(map(float, args.t))
    nvals = list(map(int, args.n))
    alpha = float(args.alpha)

    for n in nvals:
        if args.n_jobs <= 1:
            # Serial execution over reps (baseline behavior).
            for rep in range(args.M):
                rows.extend(
                    run_one_replicate(
                        rep=rep,
                        n=n,
                        alpha=alpha,
                        base=args.base,
                        tvals=tvals,
                        L=args.L,
                        level=args.level,
                        seed=args.seed,
                        backend=args.backend,
                    )
                )
        else:
            # Parallel execution over reps using ProcessPoolExecutor.
            tasks = [
                (
                    rep,
                    n,
                    alpha,
                    args.base,
                    tvals,
                    args.L,
                    args.level,
                    args.seed,
                    args.backend,
                )
                for rep in range(args.M)
            ]
            with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
                for rows_rep in ex.map(_worker_run_one_replicate, tasks):
                    rows.extend(rows_rep)

    # Persist results
    df = pd.DataFrame(rows)
    stem = f"prop26_M{args.M}_L{args.L}_a{alpha}_seed{args.seed}_{args.base}.csv"
    out = outdir / stem
    df.to_csv(out, index=False)
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
