.PHONY: figures validate enhancements all

all: figures validate enhancements

figures:
	python scripts/generate_radio_snr_sweep.py
	python scripts/radio_snr_from_qft.py
	python scripts/sweep_phase_jitter.py
	python scripts/sweep_shapes.py

validate:
	python scripts/script_validate_frequency_gating.py

enhancements:
	python -c "from multi_mirror_configurations import calculate_multi_mirror_enhancement_factors as f; print(f())"
