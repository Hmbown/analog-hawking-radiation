# Analog Hawking Radiation - Makefile
# This Makefile provides convenient shortcuts to the ahr CLI
# All commands are thin wrappers around 'ahr' for consistency

# --- Core Workflows (ahr CLI wrappers) ---
.PHONY: quickstart
quickstart:
	@ahr quickstart --out results/quickstart

.PHONY: validate
validate:
	@ahr validate

.PHONY: validate-dashboard
validate-dashboard:
	@ahr validate --dashboard

.PHONY: bench
bench:
	@ahr bench

.PHONY: info
info:
	@ahr info

.PHONY: docs
-docs:
	@ahr docs

.PHONY: tutorial
tutorial:
	@ahr tutorial --list

# --- Pipeline Execution ---
.PHONY: pipeline-demo
pipeline-demo:
	@ahr pipeline --demo --out results/pipeline_demo

.PHONY: pipeline-safe
pipeline-safe:
	@ahr pipeline --demo --safe --out results/pipeline_safe

# --- Parameter Sweeps ---
.PHONY: sweep-gradient
sweep-gradient:
	@ahr sweep --gradient --output results/gradient_catastrophe

.PHONY: sweep-gradient-full
sweep-gradient-full:
	@ahr sweep --gradient --n-samples 500 --output results/gradient_limits_production

# --- Analysis Tools ---
.PHONY: analyze-correlation
analyze-correlation:
	@ahr analyze --correlation --output figures/correlation_map.png

# --- Experiment Planning ---
.PHONY: experiment-eli
experiment-eli:
	@ahr experiment --eli --output results/eli_planning

# --- Development ---
.PHONY: dev-setup
dev-setup:
	@ahr dev --setup

# --- Comprehensive Analysis Bundle ---
.PHONY: comprehensive
comprehensive:
	@echo "Running comprehensive analysis pipeline..."
	@ahr pipeline --demo --kappa-method acoustic_exact --graybody acoustic_wkb
	@ahr sweep --gradient --n-samples 100 --output results/gradient_catastrophe
	@echo "Comprehensive analysis complete. Check results/ directory."

.PHONY: results-pack
results-pack:
	@echo "Building results package..."
	python3 scripts/build_results_pack.py
	@echo "Results package created: results/results_pack.zip"

# --- Legacy Commands (maintained for backward compatibility) ---
.PHONY: figs
figs:
	rm -rf figures

.PHONY: demo-bundle
demo-bundle:
	python3 scripts/make_demo_bundle.py

.PHONY: detection-feasibility
detection-feasibility:
	python3 scripts/standalone_detection_feasibility_demo.py

.PHONY: detection-feasibility-full
detection-feasibility-full:
	python3 scripts/comprehensive_detection_feasibility_analysis.py --n-configs 50 --generate-report

.PHONY: orchestrate
orchestrate:
	python3 -m scripts.orchestration_engine --config configs/orchestration/base.yml $(if $(NAME),--name $(NAME),) $(if $(PHASES),--phases $(PHASES),)

.PHONY: aggregate
aggregate:
	python3 scripts/result_aggregator.py $(EXPID)

.PHONY: report
report:
	python3 scripts/reporting/integration.py $(EXPID) $(if $(COMPONENT),--component $(COMPONENT),)

.PHONY: nd-demo
nd-demo:
	python3 scripts/run_horizon_nd_demo.py --dim 2 --nx 160 --ny 40 --sigma 4e-7 --v0 2.0e6 --cs0 1.0e6 --x0 5e-6

.PHONY: nd-pic-npz
nd-pic-npz:
	python3 scripts/run_pic_pipeline.py --nd-npz results/grid_nd_profile.npz --nd-scan-axis 0 --graybody acoustic_wkb --alpha-gray 1.0 --B 1e8 --Tsys 30

.PHONY: nd-pic-h5
nd-pic-h5:
	python3 scripts/run_pic_pipeline.py --nd-h5-in $(H5) --x-ds $(XDS) --y-ds $(YDS) $(if $(ZDS),--z-ds $(ZDS),) --vx-ds $(VXDS) --vy-ds $(VYDS) $(if $(VZDS),--vz-ds $(VZDS),) --cs-ds $(CSDS) --nd-scan-axis 0 --nd-patches 64 --graybody acoustic_wkb --alpha-gray 1.0 --B 1e8 --Tsys 30

.PHONY: nd-generate-h5
nd-generate-h5:
	python3 scripts/generate_synthetic_nd_h5.py --out results/synthetic_nd.h5 --dim 2

.PHONY: nd-test
nd-test:
	pytest -q tests/test_horizon_nd.py tests/test_nd_aggregator.py

.PHONY: sweep-thresholds
sweep-thresholds:
	python3 scripts/sweep_kappa_thresholds.py --n-samples 60 --v-fracs 0.4,0.5,0.6 --dv-max 2e12,4e12,8e12 --intensity-max 1e24 --out results/threshold_sensitivity.json

# --- Quality Assurance ---
.PHONY: lint
lint:
	pre-commit run --all-files

.PHONY: test
-test:
	pytest -q

.PHONY: test-full
test-full:
	pytest -v

# --- Documentation ---
.PHONY: docs-serve
docs-serve:
	mkdocs serve -a 127.0.0.1:8000

.PHONY: docs-build
docs-build:
	mkdocs build --strict

# --- Docker ---
.PHONY: docker-build-cpu
docker-build-cpu:
	docker build -t ahr:cpu -f Dockerfile.cpu .

.PHONY: docker-run-cpu
docker-run-cpu:
	docker run --rm -it -v $$(pwd):/data -w /data ahr:cpu ahr quickstart --out /data/results/docker_cpu

.PHONY: docker-build-cuda
docker-build-cuda:
	docker build -t ahr:cuda -f Dockerfile.cuda .

.PHONY: docker-run-cuda
docker-run-cuda:
	docker run --gpus all --rm -it -v $$(pwd):/data -w /data ahr:cuda ahr quickstart --out /data/results/docker_cuda

.PHONY: bench-suite
bench-suite:
	python3 scripts/benchmarks/bench_kernels.py | tee results/bench_horizon.json

# --- Help ---
.PHONY: help
help:
	@echo "Analog Hawking Radiation - Make Targets"
	@echo "========================================"
	@echo ""
	@echo "Core Workflows (ahr CLI):"
	@echo "  make quickstart        # Quickstart demo"
	@echo "  make validate          # Run validation suite"
	@echo "  make validate-dashboard # Validation with dashboard"
	@echo "  make bench             # Benchmark kernels"
	@echo "  make info              # Show system info"
	@echo "  make docs              # Open documentation"
	@echo "  make tutorial          # List tutorials"
	@echo ""
	@echo "Pipelines:"
	@echo "  make pipeline-demo     # Demo pipeline"
	@echo "  make pipeline-safe     # Conservative demo"
	@echo ""
	@echo "Sweeps & Analysis:"
	@echo "  make sweep-gradient    # Gradient catastrophe sweep"
	@echo "  make analyze-correlation # Correlation analysis"
	@echo ""
	@echo "Experiment Planning:"
	@echo "  make experiment-eli    # ELI facility planning"
	@echo ""
	@echo "Development:"
	@echo "  make dev-setup         # Set up dev environment"
	@echo "  make lint              # Run linting"
	@echo "  make test              # Run tests"
	@echo ""
	@echo "Comprehensive:"
	@echo "  make comprehensive     # Full analysis suite"
	@echo "  make results-pack      # Package results"
	@echo ""
	@echo "For more commands, see: ahr --help"

# --- Default Target ---
.DEFAULT_GOAL := help
