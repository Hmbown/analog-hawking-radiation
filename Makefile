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

.PHONY: dashboard
dashboard:
	# Live dashboard for a given experiment ID: make dashboard EXPID=abcd1234
	python3 scripts/monitoring/dashboard.py $(EXPID)

.PHONY: aggregate
aggregate:
	# Aggregate results and generate a report: make aggregate EXPID=abcd1234
	python3 scripts/result_aggregator.py $(EXPID)

.PHONY: report
report:
	# Perform reporting integration across components: make report EXPID=abcd1234 COMPONENT=all
	python3 scripts/reporting/integration.py $(EXPID) $(if $(COMPONENT),--component $(COMPONENT),)
