.PHONY: figures validate enhancements all

all: figures validate enhancements

figures:
	python3 scripts/generate_radio_snr_sweep.py
	python3 scripts/radio_snr_from_qft.py
	python3 scripts/sweep_phase_jitter.py
	python3 scripts/sweep_shapes.py

# Generate three curated figures for README and copy to docs/img
.PHONY: hero-images
hero-images:
	python3 scripts/compare_hybrid_apples_to_apples.py
	python3 scripts/sweep_hybrid_params.py
	python3 scripts/radio_snr_from_qft.py
	mkdir -p docs/img
	@if [ -f figures/hybrid_apples_to_apples.png ]; then cp -f figures/hybrid_apples_to_apples.png docs/img/; fi
	@if [ -f figures/hybrid_t5_ratio_map.png ]; then cp -f figures/hybrid_t5_ratio_map.png docs/img/; fi
	@if [ -f figures/radio_snr_from_qft.png ]; then cp -f figures/radio_snr_from_qft.png docs/img/; fi

# Generate all essential figures for public README (comprehensive set)
.PHONY: readme-images
readme-images:
	@echo "Generating all README figures..."
	python3 scripts/generate_workflow_diagram.py
	python3 scripts/compare_hybrid_apples_to_apples.py
	python3 scripts/sweep_hybrid_params.py
	python3 scripts/compute_formation_frontier.py
	python3 scripts/radio_snr_from_qft.py
	python3 scripts/geometry_optimize_kappa.py
	python3 scripts/monte_carlo_horizon_uncertainty.py
	python3 scripts/generate_guidance_map.py
	python3 scripts/sweep_shapes.py
	@echo "Copying figures to docs/img/..."
	mkdir -p docs/img
	@if [ -f figures/workflow_diagram.png ]; then cp -f figures/workflow_diagram.png docs/img/; echo "  ✓ workflow_diagram.png"; fi
	@if [ -f figures/hybrid_apples_to_apples.png ]; then cp -f figures/hybrid_apples_to_apples.png docs/img/; echo "  ✓ hybrid_apples_to_apples.png"; fi
	@if [ -f figures/hybrid_t5_ratio_map.png ]; then cp -f figures/hybrid_t5_ratio_map.png docs/img/; echo "  ✓ hybrid_t5_ratio_map.png"; fi
	@if [ -f figures/formation_frontier.png ]; then cp -f figures/formation_frontier.png docs/img/; echo "  ✓ formation_frontier.png"; fi
	@if [ -f figures/radio_snr_from_qft.png ]; then cp -f figures/radio_snr_from_qft.png docs/img/; echo "  ✓ radio_snr_from_qft.png"; fi
	@if [ -f figures/graybody_impact.png ]; then cp -f figures/graybody_impact.png docs/img/; echo "  ✓ graybody_impact.png"; fi
	@if [ -f figures/geometry_vs_kappa.png ]; then cp -f figures/geometry_vs_kappa.png docs/img/; echo "  ✓ geometry_vs_kappa.png"; fi
	@if [ -f figures/horizon_probability_bands.png ]; then cp -f figures/horizon_probability_bands.png docs/img/; echo "  ✓ horizon_probability_bands.png"; fi
	@if [ -f figures/bayesian_guidance_map.png ]; then cp -f figures/bayesian_guidance_map.png docs/img/; echo "  ✓ bayesian_guidance_map.png"; fi
	@if [ -f figures/cs_profile_impact.png ]; then cp -f figures/cs_profile_impact.png docs/img/; echo "  ✓ cs_profile_impact.png"; fi
	@if [ -f figures/enhancement_bar.png ]; then cp -f figures/enhancement_bar.png docs/img/; echo "  ✓ enhancement_bar.png"; fi
	@if [ -f figures/optimal_glow_parameters.png ]; then cp -f figures/optimal_glow_parameters.png docs/img/; echo "  ✓ optimal_glow_parameters.png"; fi
	@echo "✓ All README images prepared in docs/img/"

# Verify all README image references exist
.PHONY: check-readme-images
check-readme-images:
	@echo "Checking README image references..."
	@for img in workflow_diagram hybrid_apples_to_apples hybrid_t5_ratio_map formation_frontier radio_snr_from_qft graybody_impact geometry_vs_kappa horizon_probability_bands bayesian_guidance_map cs_profile_impact enhancement_bar optimal_glow_parameters; do \
		if [ -f "docs/img/$$img.png" ]; then \
			echo "  ✓ $$img.png"; \
		else \
			echo "  ✗ MISSING: $$img.png"; \
		fi \
	done

validate:
	python3 scripts/script_validate_frequency_gating.py

enhancements:
	python3 -c "import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts/archive_exploratory'); from multi_mirror_configurations import calculate_multi_mirror_enhancement_factors as f; print(f())"

hybrid:
	python3 scripts/compare_hybrid_apples_to_apples.py
	python3 scripts/sweep_hybrid_params.py

# Clean build artifacts and logs (does not touch sources)
.PHONY: clean-build
clean-build:
	rm -rf paper/build_arxiv
	rm -f firebase-debug.log

# Additional housekeeping (safe to run; does not touch sources)
.PHONY: clean-results clean-figures
clean-results:
	rm -rf results

clean-figures:
	rm -rf figures
