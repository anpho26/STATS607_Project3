# =========================
# Stats 607 Project Makefile
# =========================
SHELL := /bin/bash
.ONESHELL:

# ---- Global knobs ----
PY      := python
SEED    := 2025
BASE    := uniform
ALPHA   := 5.0
N       := 1000
TVALS   := 0.25 0.5 0.75

# Part A panel ns (include 0 for the "prior" panel)
PARTA_NS := 0 100 500 1000
# Part A prior M values (for n=0 to show convergence)
PRIOR_M_VALUES := 10 100 1000 4000

# Output dirs
RAW_DIR      := results/raw
FIG_DIR      := results/figures
SUMMARY_DIR  := results/summary
PROFILE_DIR  := results/profile

# Convenience stem used by Part B figures/logs
PARTB_STEM := partB_n$(N)_a$(ALPHA)_seed$(SEED)_$(BASE)

.PHONY: all simulate analyze figures clean test help everything

# Full pipeline per rubric
all: clean simulate analyze figures

# old alias stays
everything: all

# ------------------------
# Simulate: generate RAW outputs only
# ------------------------
simulate:
	@mkdir -p $(RAW_DIR)
	@echo "[simulate] Part B: distances + Pm paths"
	$(PY) -m src_cli.partb_log_convergence --n 1000 --alpha $(ALPHA) --t $(TVALS) --seed $(SEED) --base $(BASE)
	@echo "[simulate] Part C: prop 2.6 pooled-Z (logs)"
	$(PY) -m src_cli.partc_log_prop26 --alpha $(ALPHA) --t $(TVALS) --n 100 500 1000 --M 400 --seed $(SEED) --base $(BASE)

# ------------------------
# Analyze: summarize raw results (tidy CSVs)
# ------------------------
analyze: simulate
	@mkdir -p $(SUMMARY_DIR)
	@echo "[analyze] summarizing -> $(SUMMARY_DIR)"
	$(PY) -m src_cli.analyze --raw $(RAW_DIR) --summary $(SUMMARY_DIR) --alpha $(ALPHA) --seed $(SEED) --base $(BASE) --stem $(PARTB_STEM)

# ------------------------
# Figures: produce ALL required plots
# ------------------------
figures:
	@mkdir -p $(FIG_DIR)
	@echo "[figures] Part A prior panels (n=0, varying M)"
	for m in $(PRIOR_M_VALUES); do \
	  $(PY) -m src_cli.parta_panels --base $(BASE) --t $(TVALS) --alpha 1 5 20 --n 0 --M $$m --N 2000 --seed $(SEED); \
	done
	@echo "[figures] Part A posterior panels (n=100,500,1000)"
	for n in 100 500 1000; do \
	  $(PY) -m src_cli.parta_panels --base $(BASE) --t $(TVALS) --alpha 1 5 20 --n $$n --M 4000 --N 2000 --seed $(SEED); \
	done
	@echo "[figures] Part B (convergence, predictive paths)"
	$(PY) -m src_cli.partb_figures --stem $(PARTB_STEM) --title "n=1000, α=$(ALPHA), base=$(BASE)"
	@echo "[figures] Part C (pooled Z only)"
	$(PY) -m src_cli.partc_figures_prop26 --csv $(RAW_DIR)/prop26_M400_L50000_a$(ALPHA)_seed$(SEED)_$(BASE).csv --title "Proposition 2.6: α=$(ALPHA), base=$(BASE)"

# ------------------------
# Clean: remove generated artifacts
# ------------------------
clean:
	@echo "[clean] removing results and caches"
	@rm -rf $(RAW_DIR)* $(FIG_DIR)* $(SUMMARY_DIR)* .pytest_cache __pycache__
	@mkdir -p $(RAW_DIR) $(FIG_DIR) $(SUMMARY_DIR)

# ------------------------
# Test suite
# ------------------------
test:
	pytest -q

# ------------------------
# Help
# ------------------------
help:
	@echo "Targets:"
	@echo "  make all       - clean → simulate → analyze → figures (full pipeline)"
	@echo "  make simulate  - run simulations and save RAW outputs"
	@echo "  make analyze   - process RAW outputs into summary CSVs"
	@echo "  make figures   - create all visualizations"
	@echo "  make clean     - remove generated files"
	@echo "  make test      - run pytest"
	@echo "  make help      - this message"
	@echo "  make partA-prior - Part A prior panels (n=0, M=10,100,1000,4000)"
	@echo "  make partA       - Part A posterior panels (n=100,500,1000)"
	@echo "  make partB       - Part B logs + figures"
	@echo "  make partC       - Part C logs + pooled-Z figure"

# ------------------------
# Individual part targets
# ------------------------
.PHONY: partA-prior
partA-prior:
	@mkdir -p $(FIG_DIR)
	for m in $(PRIOR_M_VALUES); do \
	  $(PY) -m src_cli.parta_panels --base $(BASE) --t $(TVALS) --alpha 1 5 20 --n 0 --M $$m --N 2000 --seed $(SEED); \
	done

.PHONY: partA
partA:
	@mkdir -p $(FIG_DIR)
	for n in 100 500 1000; do \
	  $(PY) -m src_cli.parta_panels --base $(BASE) --t $(TVALS) --alpha 1 5 20 --n $$n --M 4000 --N 2000 --seed $(SEED); \
	done

.PHONY: partB
# Part B only (logs + figures)
partB:
	@mkdir -p $(RAW_DIR) $(FIG_DIR)
	$(PY) -m src_cli.partb_log_convergence --n $(N) --alpha $(ALPHA) --t $(TVALS) --seed $(SEED) --base $(BASE)
	$(PY) -m src_cli.partb_figures --stem $(PARTB_STEM) --title "n=$(N), α=$(ALPHA), base=$(BASE)"

.PHONY: partC
partC:
	@mkdir -p $(RAW_DIR) $(FIG_DIR)
	$(PY) -m src_cli.partc_log_prop26 --alpha $(ALPHA) --t $(TVALS) --n 100 500 1000 --M 400 --seed $(SEED) --base $(BASE)
	$(PY) -m src_cli.partc_figures_prop26 --csv $(RAW_DIR)/prop26_M400_L50000_a$(ALPHA)_seed$(SEED)_$(BASE).csv --title "Proposition 2.6: α=$(ALPHA), base=$(BASE)"



# =====================
# Unit 3 extra targets
# =====================

.PHONY: profile complexity benchmark parallel stability-check

# Global profiling directory
PROFILE_DIR := results/profile

.PHONY: profile
profile:
	@mkdir -p $(PROFILE_DIR)
	# Baseline Part A: prior (n=0) and posterior (n=1000) panels
	$(PY) -m cProfile -o $(PROFILE_DIR)/parta_prior_baseline.prof \
	  -m src_cli.parta_panels --base $(BASE) --t $(TVALS) \
	  --alpha 1 5 20 --n 0 --M 4000 --N 2000 --seed $(SEED) --backend baseline
	$(PY) -m cProfile -o $(PROFILE_DIR)/parta_post_baseline.prof \
	  -m src_cli.parta_panels --base $(BASE) --t $(TVALS) \
	  --alpha 1 5 20 --n 1000 --M 4000 --N 2000 --seed $(SEED) --backend baseline
	# Baseline Part B: convergence logging
	$(PY) -m cProfile -o $(PROFILE_DIR)/partb_log_convergence.prof \
	  -m src_cli.partb_log_convergence --n $(N) --alpha $(ALPHA) \
	  --t $(TVALS) --seed $(SEED) --base $(BASE)
	# Baseline Part C: Monte Carlo study (single job)
	$(PY) -m cProfile -o $(PROFILE_DIR)/partc_log_prop26_baseline.prof \
	  -m src_cli.partc_log_prop26 --alpha $(ALPHA) --t $(TVALS) \
	  --n 100 500 1000 --M 400 --seed $(SEED) --base $(BASE) \
	  --backend baseline --n-jobs 1
	# Fast Part A: prior and posterior panels
	$(PY) -m cProfile -o $(PROFILE_DIR)/parta_prior_fast.prof \
	  -m src_cli.parta_panels --base $(BASE) --t $(TVALS) \
	  --alpha 1 5 20 --n 0 --M 4000 --N 2000 --seed $(SEED) --backend fast
	$(PY) -m cProfile -o $(PROFILE_DIR)/parta_post_fast.prof \
	  -m src_cli.parta_panels --base $(BASE) --t $(TVALS) \
	  --alpha 1 5 20 --n 1000 --M 4000 --N 2000 --seed $(SEED) --backend fast
	# Fast Part C: Monte Carlo study (single job to avoid multiprocessing)
	$(PY) -m cProfile -o $(PROFILE_DIR)/partc_log_prop26_fast.prof \
	  -m src_cli.partc_log_prop26 --alpha $(ALPHA) --t $(TVALS) \
	  --n 100 500 1000 --M 400 --seed $(SEED) --base $(BASE) \
	  --backend fast --n-jobs 1

# Run computational complexity analysis (timing vs n / M) for part A
complexity:
	@mkdir -p results/summary
	$(PY) scripts/complexity_partA.py --backends baseline fast

benchmark:
	@echo "[benchmark] Part A: baseline vs fast backend"
	$(PY) scripts/time_parta_backend.py
	@echo "[benchmark] Part C: baseline vs fast backend at M=400"
	$(PY) scripts/complexity_partC.py --mode complexity --backends baseline fast --M 400
	@echo "[benchmark] Part C speedup analysis (fast backend with varying workers)"
	$(PY) scripts/complexity_partC.py --mode speedup --M 400 --n 100 500 1000 \
	  --alpha $(ALPHA) --base $(BASE) --t $(TVALS) --seed $(SEED) \
	  --n-jobs-list 1 2 4

# Run optimized version with parallelization (Part C fast backend, multiple workers)
parallel:
	@echo "[parallel] Part C fast backend with parallel jobs"
	$(PY) scripts/complexity_partC.py \
	  --mode speedup \
	  --M 400 \
	  --n 100 500 1000 \
	  --alpha $(ALPHA) \
	  --base $(BASE) \
	  --t $(TVALS) \
	  --seed $(SEED) \
	  --n-jobs-list 1 2 4

# Check for stability / correctness across conditions
# (runs regression tests: baseline vs fast, plus basic sanity checks)
stability-check:
	$(PY) -m scripts.test_regression


# ------------------------
# Benchmarking
# ------------------------
TIME ?= /usr/bin/time -p

.PHONY: bench-baseline bench-optimized

# Baseline: normal pipeline
bench-baseline:
	@echo "[bench-baseline] timing baseline full pipeline (make all)"
	@$(TIME) $(MAKE) clean all

# Optimized: same pipeline, but with OPTIMIZED=1 in the environment
bench-optimized:
	@echo "[bench-optimized] timing optimized full pipeline (make all)"
	@OPTIMIZED=1 $(TIME) $(MAKE) clean all