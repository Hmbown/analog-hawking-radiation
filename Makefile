.PHONY: figs
figs:
	rm -rf figures

.PHONY: demo-bundle
demo-bundle:
	python3 scripts/make_demo_bundle.py

# --- Orchestration & Reporting ---
.PHONY: orchestrate
orchestrate:
	# Run multi-phase orchestration (override PHASES="phase_1_initial_exploration ..." as needed)
	python3 -m scripts.orchestration_engine --config configs/orchestration/base.yml $(if $(NAME),--name $(NAME),) $(if $(PHASES),--phases $(PHASES),)


.PHONY: aggregate
aggregate:
	# Aggregate results and generate a report: make aggregate EXPID=abcd1234
	python3 scripts/result_aggregator.py $(EXPID)

.PHONY: report
report:
	# Perform reporting integration across components: make report EXPID=abcd1234 COMPONENT=all
	python3 scripts/reporting/integration.py $(EXPID) $(if $(COMPONENT),--component $(COMPONENT),)

# --- nD Utilities ---
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

# --- Comprehensive analysis bundle ---
.PHONY: comprehensive
comprehensive:
	python3 comprehensive_analysis.py
	python3 optimization_analysis.py

.PHONY: results-pack
results-pack:
	# Build a shareable ZIP with figures, data, and a summary
	python3 scripts/build_results_pack.py
