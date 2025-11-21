# Stats 607 — Project 2 (Pólya/DP)

This repository contains a **reproducible pipeline** implementing and extending simulation experiments from **Fortini & Petrone (2023, 2024)** [[1](#ref1), [2](#ref2)] to demonstrate how Bayesian uncertainty quantification can be expressed entirely through predictive rules under exchangeability.

## Project Overview

We investigate three core aspects of Bayesian prediction in Pólya sequences:

- **Part A (Prior/Posterior Panels)**: Visualize how predictive distributions $P_m(t)$ evolve as data accumulate, showing convergence from prior to posterior for different concentrations$\alpha$and thresholds$t$.

- **Part B (Convergence Diagnostics)**: Assess how rapidly the predictive distribution$P_m$converges to the directing random measure $\tilde{P}$ via grid-based distance metrics (sup-norm $d^{(\infty)}$, RMSE) and visualize predictive path trajectories.

- **Part C (Asymptotic Validation)**: Verify **Proposition 2.6** from Fortini & Petrone (2024) [[2](#ref2)], which provides a Gaussian approximation of the posterior distribution:
 $$\tilde{P}(A) \mid x_{1:n} \approx \mathcal{N}\left(P_n(A), \frac{V_{n,A}}{n}\right)$$
  We test this by computing pooled-Z statistics and checking their convergence to $\mathcal{N}(0,1)$.

All figures are saved to `results/figures` as **PNG**, and as **PDF** for print‑quality.

---

## References

<a name="ref1"></a>[1] Fortini, S., & Petrone, S. (2023). Prediction-based uncertainty quantification for exchangeable sequences. *Philosophical Transactions of the Royal Society A*, 381(2247), 20220142. [https://doi.org/10.1098/rsta.2022.0142](https://doi.org/10.1098/rsta.2022.0142)

<a name="ref2"></a>[2] Fortini, S., & Petrone, S. (2024). Exchangeability, prediction and predictive modeling in Bayesian statistics. arXiv preprint arXiv:2402.10126v2.

---

## Quick start
Requires Python >= 3.12
```bash
python -m venv .venv && source .venv/bin/activate #or python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Full pipeline: clean → simulate → analyze → figures
make all
```

Useful single steps:

```bash
make partA-prior   # Part A prior panels (n=0, M=10,100,1000,4000)
make partA         # Part A posterior panels (n=100, 500, 1000)
make partB         # Part B logs + figures (convergence & predictive paths)
make partC         # Part C logs + pooled‑Z figure

make simulate      # just raw logs (Part B + Part C)
make analyze       # summaries → results/summary/
make figures       # all figures (A/B/C) from existing logs

make clean         # remove generated files
make test          # run pytest
make help          # list targets
```

You can override defaults at invocation time:

```bash
make all BASE=normal ALPHA=10 SEED=7
```

Default knobs: `BASE=uniform`, `ALPHA=5.0`, `SEED=2025`, thresholds `T=0.25 0.5 0.75`.

---

## Outputs

**Raw logs** (`results/raw/`)
- `distances_partB_n<N>_a<ALPHA>_seed<SEED>_<BASE>.csv` — grid‑based distances vs stream index.
- `Pm_paths_partB_n<N>_a<ALPHA>_seed<SEED>_<BASE>.csv` — predictive path values `P_m(t)`.
- `prop26_M400_L50000_a<ALPHA>_seed<SEED>_<BASE>.csv` — pooled‑Z inputs across `n ∈ {100,500,1000}`.

**Summaries** (`results/summary/`)
- `prop26_summary.csv` — coverage & mean width by `(n, t)`.
- `partB_distances_summary.csv` — mean/min/max for `d^{(∞)}` and RMSE (if distances CSV present).

**Figures** (`results/figures/`)
- **Part A**: 
  - Prior panels: `post_panels_cont_n0_M{10,100,1000,4000}_N2000_<base>.png` (showing convergence with increasing M)
  - Posterior panels: `post_panels_cont_n{100,500,1000}_M4000_N2000_<base>.png`
- **Part B**: `partB_convergence_<stem>.(png|pdf)`, `partB_paths_<stem>.(png|pdf)`
- **Part C**: `prop26_zcheck_<stem>.(png|pdf)` (histogram overlaid with `N(0,1)` density)

> PDF export is enabled for all Part B/C figures and used in Part A wherever safe. Rendering uses the project style in `src/plotstyle.py` (set via `apply_plot_style()` in each CLI).

---

## Repository structure (annotated)

```
.
├─ config/                      # optional configuration/snippets for experiments
├─ examples/                    # example notebooks or small demo scripts (optional)
├─ figures/                     # (optional) manually curated figures
├─ results/
│  ├─ raw/                      # raw CSV logs written by CLI "log_*" scripts
│  ├─ figures/                  # auto‑generated figures (PNG/PDF)
│  └─ summary/                  # tidy CSV summaries created by `make analyze`
├─ scripts/                     # helper shell/python scripts (not required by pipeline)
├─ src/                         # core library code (importable as `src.*`)
│  ├─ dgps.py                   # data‑generating processes (Uniform, Normal, …)
│  ├─ interfaces.py             # minimal protocol/typing helpers
│  ├─ methods.py                # Pólya predictive implementation wrappers
│  ├─ metrics.py                # grids, distances (d_infty, RMSE), helpers
│  ├─ plotstyle.py              # `apply_plot_style()` for research‑quality plots
│  ├─ polya.py                  # Pólya sequence model utilities
│  └─ simulation.py             # simulation helpers used by CLIs/tests
├─ src_cli/                     # command‑line entry points (callable via `python -m ...`)
│  ├─ analyze.py                # aggregates raw → summary CSVs
│  ├─ parta_panels.py           # generate Part A panels for n in {0,100,500,1000}
│  ├─ partb_log_convergence.py  # log Part B distances + Pm paths
│  ├─ partb_figures.py          # render Part B figures from logs
│  ├─ partc_log_prop26.py       # log data for Proposition 2.6 pooled‑Z
│  └─ partc_figures_prop26.py   # plot pooled‑Z histogram + N(0,1) overlay
├─ tests/                       # pytest suite
│  ├─ conftest.py
│  ├─ test_dgp.py               # DGP sanity checks
│  ├─ test_exchangeability.py   # basic exchangeability/symmetry checks
│  ├─ test_polya_module.py      # Pólya utilities
│  ├─ test_repro.py             # reproducibility (seeded runs)
│  ├─ test_smoke_parta.py       # smoke test Part A CLIs
│  ├─ test_smoke_partb.py       # smoke test Part B CLIs
│  └─ test_smoke_partc.py       # smoke test Part C CLIs
├─ Makefile                     # orchestration: all / simulate / analyze / figures / clean / test
├─ README.md
├─ requirements.txt             # Python dependencies
└─ pytest.ini                   # pytest defaults
```

---

## Command‑line interfaces (CLIs)

### Part A — Panels
```bash
# Prior panels (n=0) with varying M to show convergence
python -m src_cli.parta_panels --base uniform --t 0.25 0.5 0.75 --alpha 1 5 20 --n 0 --M 10 --N 2000 --seed 2025
python -m src_cli.parta_panels --base uniform --t 0.25 0.5 0.75 --alpha 1 5 20 --n 0 --M 100 --N 2000 --seed 2025
python -m src_cli.parta_panels --base uniform --t 0.25 0.5 0.75 --alpha 1 5 20 --n 0 --M 1000 --N 2000 --seed 2025
python -m src_cli.parta_panels --base uniform --t 0.25 0.5 0.75 --alpha 1 5 20 --n 0 --M 4000 --N 2000 --seed 2025

# Posterior panels (n>0)
python -m src_cli.parta_panels --base uniform --t 0.25 0.5 0.75 --alpha 1 5 20 --n 100 --M 4000 --N 2000 --seed 2025
python -m src_cli.parta_panels --base uniform --t 0.25 0.5 0.75 --alpha 1 5 20 --n 500 --M 4000 --N 2000 --seed 2025
python -m src_cli.parta_panels --base uniform --t 0.25 0.5 0.75 --alpha 1 5 20 --n 1000 --M 4000 --N 2000 --seed 2025
```

### Part B — Convergence + Predictive Paths 
```bash
python -m src_cli.partb_log_convergence --n 1000 --alpha 5.0 --t 0.25 0.5 0.75 --seed 2025 --base uniform

python -m src_cli.partb_figures --stem partB_n1000_a5.0_seed2025_uniform --title "n=1000, α=5.0, base=uniform"
```

### Part C — Proposition 2.6 
```bash
python -m src_cli.partc_log_prop26 --alpha 5.0 --t 0.25 0.5 0.75 --n 100 500 1000 --M 400 --seed 2025 --base uniform

python -m src_cli.partc_figures_prop26 --csv results/raw/prop26_M400_L50000_a5.0_seed2025_uniform.csv --title "Proposition 2.6: α=5.0, base=uniform"
```

### Analyze (summaries)
```bash
python -m src_cli.analyze --raw results/raw --summary results/summary --alpha 5.0 --seed 2025 --base uniform --stem partB_n1000_a5.0_seed2025_uniform
```

---

## Notes on reproducibility and style

- All CLIs accept `--seed`; the Makefile uses `SEED=2025` by default.
- Plotting calls `apply_plot_style()` which sets readable fonts, tight bounding boxes, and PDF‑friendly rcParams (TrueType embedding).
- Figures are deterministic given the seed and parameters.

---

## Estimated runtime

~7 minutes for the full pipeline (`make all`) on a typical laptop.

## Summary of key findings
Across settings, the one-step predictive from the Pólya/DP model is well-calibrated: the pooled \(Z\) statistic closely follows \(\mathcal{N}(0,1)\) (Prop. 2.6). Distances between the predictive CDF and the oracle shrink with \(n\), and the influence of \(\alpha\) is strongest early but fades as data accumulate.

---

## Division of Work

### Project Context
This project reproduces and extends the simulation experiments in Fortini & Petrone (2023). We jointly designed the study, selected parameter grids, and verified results. In accordance with the assignment requirements, *each partner submits an individual `ANALYSIS.md`* with their own critical perspective.

### An Pho
**Focus:** Predictive Simulation & Diagnostics
- Implemented core simulation modules under `src/`: `dgps.py`, `methods.py`, `polya.py`, `simulation.py`.
- Developed CLI scripts for predictive learning and diagnostics: `src_cli/parta_panels.py`, `src_cli/partb_log_convergence.py`, `src_cli/partb_figures.py`.
- Authored `ADEMP.md`, created Makefile automation, and produced Part A (prior/posterior) and Part B (predictive convergence) figures under `results/figures/`.
- Contributed tests for DGP sanity checks, exchangeability, and Part A/B smoke tests.

### Thanh Nguyen
**Focus:** Asymptotic Evaluation & Repository Management
- Implemented evaluation and metrics in `src/metrics.py` (RMSE, sup-norm distances, coverage, and variance terms V_{n,t}).
- Led Part C implementation and figures: `src_cli/partc_log_prop26.py`, `src_cli/partc_figures_prop26.py`; verified pooled-Z and credible-interval coverage.
- Maintained version control workflow and performed the majority of *Git pushes* to the shared GitHub repository (branching, commits, merges, remote sync).
- Extended test coverage for reproducibility and Part C smoke tests.

### Joint Responsibilities
- Designed simulation grid and agreed on random seeds.
- Validated predictive trajectories, convergence diagnostics, and reviewed summary outputs under `results/summary/`.
- Ensured consistent file structure, figure formats, and reproducibility via the `Makefile` pipeline (`make all`).
