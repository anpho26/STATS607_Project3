# Stats 607 — Project 3
This repository contains the code and reproducible pipeline for **Project 3**, extending the simulation framework developed in Project 2.  
All detailed analysis has been moved to separate documents for clarity.

## What this repository contains
- A fully automated simulation pipeline (Parts A, B, C)
- Command‑line interfaces for generating logs, summaries, and figures
- Benchmarking, profiling, and optimization scripts
- A reproducible Makefile workflow

## How to run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make all        # full pipeline: clean → simulate → analyze → figures
make partA      # Part A panels (prior + posterior)
make partB      # Part B convergence / predictive checks
make partC      # Part C logs + pooled-Z figure
make simulate   # run simulations for Parts B and C
make analyze    # raw → summary CSVs
make figures    # generate all figures

make clean      # remove generated outputs
```

```bash
# Additional Project 3 requirements:
make profile      # profiling for Parts A, B, C (baseline + optimized)
make complexity   # complexity analysis (log–log plots)
make benchmark    # overall timing comparison (baseline vs optimized)
make parallel     # Part C speedup study using multiple workers
make help         # show all available make targets
```

Default parameters can be overridden:
```bash
make all BASE=normal ALPHA=10 SEED=7
```

## Where results appear
- `results/raw/` — CSV logs  
- `results/summary/` — tidy summaries  
- `results/figures/` — PNG/PDF plots  

## Documentation
- **ANALYSIS.md** — full statistical analysis and interpretation  
- **OPTIMIZATION.md** — profiling, complexity analysis, and speedups  
- **ADEMP.md** — experiment design and structure  
- **Makefile** — complete reproducible workflow  

## Optimization Overview
This project includes an optimized computation pipeline for Parts A and C:

- **Part A:** A NumPy‑based fast backend reduces Python list‑handling overhead and achieves ~3× speedup over the baseline implementation.
- **Part C:** Parallel execution using multiple workers accelerates Monte Carlo simulation, achieving near‑linear speedup up to 4 jobs.

For a full discussion of profiling results, complexity analysis, and speedups, see **OPTIMIZATION.md**.

## Notes
- All simulations are deterministic given the seed.
- Figures use the global style defined in `src/plotstyle.py`.

For implementation details, see the source modules under `src/` and the CLI scripts under `src_cli/`.
