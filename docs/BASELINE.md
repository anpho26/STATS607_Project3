# Baseline Performance Report


This document summarizes the baseline computational performance of my Unit 2
simulation code, prior to any Unit 3 optimizations. All experiments were run on:

- **Machine:** arm64, Darwin
- **Python version:** 3.14.0
- **Environment:** Python venv at `.venv/` in project root

## Total Runtime of Complete Pipeline

Command used:

```
(time make all BASE=uniform ALPHA=5.0 SEED=2025)
```

| Metric | Value |
|--------|-------|
| Real time | 364.88 s |
| User CPU time | 360.23 s |
| System CPU time | 2.27 s |

**Interpretation:** The complete pipeline required approximately 6 minutes of wall-clock time, with almost all time spent in user-mode computation rather than system calls. The high CPU utilization (≈99%) indicates that the simulation kept the processor fully occupied throughout.

## Runtime Profiling Results

We profiled the three main components of the pipeline (Parts A, B, and C) using cProfile and visualized the results with Snakeviz.

### Part A: Prior/posterior panels (Pólya urn simulation)

```bash
# Prior panels (n = 0, M = 4000)
python -m cProfile -o results/profile/parta_prior_M4000.prof \
  -m src_cli.parta_panels \
    --base uniform --t 0.25 0.5 0.75 --alpha 1 5 20 \
    --n 0 --M 4000 --N 2000 --seed 2025

# Posterior panels (n = 1000, M = 4000)
python -m cProfile -o results/profile/parta_posterior_n1000.prof \
  -m src_cli.parta_panels \
    --base uniform --t 0.25 0.5 0.75 --alpha 1 5 20 \
    --n 1000 --M 4000 --N 2000 --seed 2025
```

```bash
snakeviz results/profile/parta_prior_M4000.prof   # representative heavy case
```

#### Top Functions by Cumulative Time (Part A)

| Rank | Function                               | Cum. Time (s) | Calls      | Notes                                                   |
|------|----------------------------------------|---------------|------------|---------------------------------------------------------|
| 1    | parta_panels.py:9 (`panel_for_n`)      | 108.6         | 1          | Wraps full Part A simulation and plotting for one \(n\) |
| 2    | polya.py:68 (`continue_urn_once`)      | 104.4         | 18,000     | Outer loop stepping the Pólya urn process               |
| 3    | polya.py:42 (`Pn`)                     | 88.1          | 71,982,000 | Core predictive-probability update (inner loop)         |
| 4    | polya.py:27 (`_rng`)                   | 3.5           | 72,860,694 | RNG helper used by urn updates                          |
| 5    | `<method 'append' of 'list' objects>`  | 3.6           | 72,175,745 | Python list bookkeeping inside the simulation loops     |

**Summary (Part A):** Part A’s runtime is almost entirely concentrated in the `panel_for_n` driver, which spends most of its time calling `continue_urn_once`, and ultimately the core predictive-probability routine `Pn`. The `Pn` function alone accounts for roughly 80% of the total runtime, reflecting the $O(M \times n)$ inner loop over Pólya–urn updates. Additional overhead comes from Python list operations and random-number generation, while imports, plotting, and I/O contribute negligibly to overall time.

### Part B: Convergence logging (core DP predictive simulation)

Profiling performed with:

```bash
python -m cProfile -o results/profile/partb_log_convergence.prof \
  -m src_cli.partb_log_convergence \
  --n 1000 --alpha 5.0 --t 0.25 0.5 0.75 --seed 2025 --base uniform
```

Visualized using Snakeviz:

```bash
snakeviz results/profile/partb_log_convergence.prof
```

#### Top Functions by Cumulative Time (Part B)

| Rank | Function                                          | Cum. Time (s) | Calls      | Notes                                  |
|------|---------------------------------------------------|---------------|------------|----------------------------------------|
| 1    | `<built-in method _imp.create_dynamic>`           | 0.256         | 103        | Module import + dynamic loading        |
| 2    | `<method 'read' of '_io.BufferedReader' objects>` | 0.081         | 630        | Reading bytecode / file loading        |
| 3    | `<built-in method marshal.loads>`                 | 0.037         | 625        | Unmarshalling Python bytecode          |
| 4    | `numpy.asarray`                                   | 0.033         | 13,007     | Array conversion overhead               |
| 5    | `methods.py:55 (cdf_est)`                         | 0.106         | 3,996      | CDF estimation for predictive curves   |

**Summary (Part B):** Part B is computationally lightweight, completing in under one second. 
Profiling shows that the majority of time is spent in Python’s import machinery 
(`_imp.create_dynamic`, file reading, and bytecode unmarshalling) rather than 
in the simulation logic itself. The only user-defined function with noticeable 
cost is `cdf_est`, which evaluates predictive CDF values but accounts for 
less than 0.1 seconds in total. Overall, no single computational bottleneck 
dominates Part B, and the workload is almost entirely I/O and import-bound.


### Part C: Proposition 2.6 Monte Carlo study

```bash
python -m cProfile -o results/profile/partc_log_prop26.prof \
  -m src_cli.partc_log_prop26 \
    --alpha 5.0 --t 0.25 0.5 0.75 \
    --n 100 500 1000 --M 400 --seed 2025 --base uniform
```

```bash
snakeviz results/profile/partc_log_prop26.prof
```

#### Top Functions by Cumulative Time (Part C)

| Rank | Function                                      | Cum. Time (s) | Calls       | Notes                                                           |
|------|-----------------------------------------------|---------------|-------------|-----------------------------------------------------------------|
| 1    | partc_log_prop26.py:42 (`draw_polya_next`)    | 65.27         | 60,640,000  | Core DP/Pólya update loop; dominates runtime                    |
| 2    | partc_log_prop26.py:56 (`main`)               | 89.37         | 1           | Driver wrapping all Monte Carlo repetitions                     |
| 3    | `<method 'append' of 'list' objects>`         | 3.23          | 60,654,934  | Python list growth within Pólya-urn loop                        |
| 4    | `<built-in method builtins.len>`              | 2.34          | 60,653,957  | Length checks inside the urn update routine                     |
| 5    | partc_log_prop26.py:7 (`G0_cdf`)              | 0.188         | 1,927,200   | Computation of base CDF for batches of predictive evaluations   |

**Summary (Part C):** Part C’s performance is dominated by the inner Pólya–urn update 
routine `draw_polya_next`, which accounts for more than 70% of total runtime across 
≈60 million calls. The next largest contributors are Python-level list operations 
(`append`, `len`), reflecting the overhead of repeatedly manipulating Python objects 
in a tight numerical loop. The base-measure CDF evaluation `G0_cdf` is inexpensive 
by comparison. Overall, Part C exhibits a clear single-core bottleneck in the 
sequential urn-update loop, consistent with an \(O(M \times n)\) complexity pattern.

## Computational Complexity Analysis

### Part A: Runtime vs $n$

Collected using:

```
python scripts/complexity_partA.py
```

| $n$ | Runtime (s) |
|-----|-------------|
| 0   | 73.964  |
| 100 | 72.557  |
| 500 | 65.478  |
| 1000| 58.149  |
| 1500| 47.158  |

**Slope of log--log fit:** $\text{slope} \approx -0.14$.  
**Interpretation:** Runtime decreases slightly with increasing $n$ because larger $n$ produces stronger concentration of the posterior Pólya–urn draws, reducing the number of effective branching events. The slope is close to zero, indicating no meaningful growth in runtime with respect to $n$.

### Part B: Runtime vs Sample Size $n$

Collected using:

```
python scripts/complexity_partB.py
```

| $n$ | Runtime (s) |
|-----|-------------|
| 200   | 0.632 |
| 500   | 0.413 |
| 1000  | 0.485 |
| 2000  | 0.777 |
| 5000  | 2.532 |
| 10000 | 8.536 |

**Slope of log--log fit:** $\text{slope} \approx 0.69$.  
**Interpretation:** Runtime grows roughly like $n^{0.69}$, which is close to linear on the log--log scale but slightly sublinear for the range of $n$ considered. For small $n$, fixed costs such as imports and setup dominate, while for larger $n$ the cost of the main simulation loop becomes more visible. Overall, the results are consistent with approximately $O(n)$ scaling for Part B's convergence logging step.

### Part C: Runtime vs $M$

Collected using:

```
python scripts/complexity_partC.py
```

| $M$ | Runtime (s) |
|-----|-------------|
| 100  | 17.295 |
| 200  | 34.270 |
| 400  | 68.565 |
| 800  | 148.961 |
| 1600 | 282.943 |

**Slope of log--log fit:** $\text{slope} \approx 1.02$.  
**Interpretation:** Runtime grows essentially linearly in $M$, as confirmed by the log--log slope of approximately 1.02. This is consistent with the fact that Part C performs $M$ independent Monte Carlo replicates, each requiring a full Pólya–urn trajectory. The nearly perfect $O(M)$ scaling reflects the absence of significant fixed overhead relative to the cost of each replicate.

## Numerical Stability Checks

Executed with:

```
PYTHONWARNINGS=default make all BASE=uniform ALPHA=5.0 SEED=2025
```

Checks performed:

- No overflow, underflow, or invalid-value warnings encountered.
- No NaNs or Inf values in any CSV under `results/raw` or `results/summary`.
- Predictive probabilities all in the interval [0,1].
- No exceptions or crashes occurred during simulation.

## Overall Summary

The baseline performance analysis shows that the majority of computational cost in the 
Unit 2 pipeline arises from Parts A and C, both dominated by tight Pólya–urn update loops 
that scale linearly in their respective parameters. Profiling reveals that the core 
bottleneck in Part A is the predictive-probability computation `Pn`, while Part C is 
dominated by the repeated calls to `draw_polya_next` across millions of Monte Carlo 
iterations. In contrast, Part B is negligible in cost, with most of its runtime attributable 
to Python import overhead rather than substantive computation.