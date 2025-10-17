.PHONY: figures validate enhancements all

all: figures validate enhancements

figures:
	python3 scripts/generate_radio_snr_sweep.py
	python3 scripts/radio_snr_from_qft.py
	python3 scripts/sweep_phase_jitter.py
	python3 scripts/sweep_shapes.py

# Generate two curated figures for README and copy to docs/img
.PHONY: hero-images
hero-images:
	python3 scripts/compare_hybrid_apples_to_apples.py
	python3 scripts/sweep_hybrid_params.py
	mkdir -p docs/img
	@if [ -f figures/hybrid_apples_to_apples.png ]; then cp -f figures/hybrid_apples_to_apples.png docs/img/; fi
	@if [ -f figures/hybrid_t5_ratio_map.png ]; then cp -f figures/hybrid_t5_ratio_map.png docs/img/; fi

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
