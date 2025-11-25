from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

# Restrict allowable base names at type-check time.
BaseName = Literal["uniform", "normal"]


@dataclass
class FastPolyaSequenceModel:
    """
    Optimized Blackwell–MacQueen Pólya urn, base G0 ∈ {Uniform(0,1), Normal(0,1)}.

    Differences from the baseline PolyaSequenceModel in `polya.py`:
      * Always holds a persistent RNG (no default_rng() inside hot loops).
      * Provides fast helpers that inline the predictive step and avoid Python
        list operations in the inner loop.

    Parameters
    ----------
    alpha : float
        Concentration parameter (α > 0). Larger α ⇒ stronger pull toward G0.
    base : {"uniform","normal"}
        Choice of base distribution G0. "uniform" uses U(0,1); "normal" uses N(0,1).
    rng : np.random.Generator | None
        Optional external RNG for reproducibility. If None, a fresh default_rng()
        is created once in __post_init__.
    """
    alpha: float = 5.0
    base: BaseName = "uniform"
    rng: np.random.Generator | None = None

    # These attributes are initialised in __post_init__ for fast access
    _base_draw: callable | None = None

    def __post_init__(self) -> None:
        # Ensure we always have a persistent RNG
        if self.rng is None:
            self.rng = np.random.default_rng()

        # Cache the appropriate base-draw function for quick reuse
        if self.base == "uniform":
            # Generator.random() returns floats in [0,1)
            self._base_draw = self.rng.random
        elif self.base == "normal":
            self._base_draw = self.rng.standard_normal
        else:
            raise ValueError(f"unknown base={self.base}")

    # --- API-compatible helpers (P0 / Pn) ---

    def P0(self) -> float:
        """Draw P0 from the base distribution G0 (prior)."""
        return float(self._base_draw())  # type: ignore[arg-type]

    def Pn(self, n: int, history: Sequence[float]) -> float:
        """
        Compatibility wrapper for the original one-step predictive.

        NOTE: This version is not used in the optimized loops below; they inline
        the same logic to avoid function-call overhead. This method exists so
        that existing code which calls Pn directly can still work.
        """
        if n != len(history):
            raise ValueError("n must equal len(history)")

        alpha = self.alpha
        r = self.rng  # type: ignore[assignment]
        w_base = alpha / (alpha + n)
        if r.random() < w_base:
            return self.P0()
        j = r.integers(0, n)
        return float(history[j])


# ----- Optimized helpers used by CLIs -----


def build_prefix_fast(n_obs: int, model: FastPolyaSequenceModel) -> np.ndarray:
    """
    Generate x_{1:n_obs} from the urn using an optimized loop.

    Returns
    -------
    np.ndarray of shape (n_obs,)
        Sequential Pólya-urn draws.
    """
    if n_obs <= 0:
        return np.empty(0, dtype=float)

    rng = model.rng  # type: ignore[assignment]
    base_draw = model._base_draw  # type: ignore[assignment]
    alpha = model.alpha

    x = np.empty(n_obs, dtype=float)
    # X1 ~ G0
    x[0] = base_draw()

    # Precompute α / (α + n) for n = 1, ..., n_obs-1
    ns = np.arange(1, n_obs, dtype=float)
    w_base = alpha / (alpha + ns)

    for m in range(1, n_obs):
        if rng.random() < w_base[m - 1]:
            x[m] = base_draw()
        else:
            j = rng.integers(0, m)
            x[m] = x[j]

    return x


def continue_urn_once_fast(
    prefix: Sequence[float], model: FastPolyaSequenceModel, M: int
) -> np.ndarray:
    """
    Fix x_{1:n} and continue to length M using the predictive, optimized version.

    Parameters
    ----------
    prefix : Sequence[float]
        Existing sequence (treated as observed history).
    model : FastPolyaSequenceModel
        Pólya urn model providing predictive draws.
    M : int
        Target total length after continuation (M ≥ len(prefix)).

    Returns
    -------
    np.ndarray
        Extended sequence of length M.
    """
    prefix_arr = np.asarray(prefix, dtype=float)
    n = prefix_arr.shape[0]
    if M < n:
        raise ValueError("M must be at least len(prefix)")

    rng = model.rng  # type: ignore[assignment]
    base_draw = model._base_draw  # type: ignore[assignment]
    alpha = model.alpha

    x = np.empty(M, dtype=float)
    x[:n] = prefix_arr

    if M == n:
        return x

    # For continuation, n runs from current length up to M-1
    ns = np.arange(n, M, dtype=float)
    w_base = alpha / (alpha + ns)

    for idx, m in enumerate(range(n, M)):
        if rng.random() < w_base[idx]:
            x[m] = base_draw()
        else:
            j = rng.integers(0, m)
            x[m] = x[j]

    return x


def sample_prior_once_fast(M: int, model: FastPolyaSequenceModel) -> np.ndarray:
    """
    Unconditional Pólya sequence of length M (from prior), optimized.

    Starts with a base draw, then proceeds with predictive updates.
    """
    if M <= 0:
        return np.empty(0, dtype=float)

    rng = model.rng  # type: ignore[assignment]
    base_draw = model._base_draw  # type: ignore[assignment]
    alpha = model.alpha

    x = np.empty(M, dtype=float)
    x[0] = base_draw()

    # n = 1, ..., M-1 for X_{n+1}
    ns = np.arange(1, M, dtype=float)
    w_base = alpha / (alpha + ns)

    for m in range(1, M):
        if rng.random() < w_base[m - 1]:
            x[m] = base_draw()
        else:
            j = rng.integers(0, m)
            x[m] = x[j]

    return x