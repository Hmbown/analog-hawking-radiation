.PHONY: figures validate enhancements all

all: figures validate enhancements

figures:
	python3 scripts/generate_radio_snr_sweep.py
	python3 scripts/radio_snr_from_qft.py
	python3 scripts/sweep_phase_jitter.py
	python3 scripts/sweep_shapes.py

validate:
	python3 scripts/script_validate_frequency_gating.py

enhancements:
	python3 -c "import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts/archive_exploratory'); from multi_mirror_configurations import calculate_multi_mirror_enhancement_factors as f; print(f())"

hybrid:
	python3 scripts/compare_hybrid_apples_to_apples.py
	python3 scripts/sweep_hybrid_params.py
