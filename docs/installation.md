# Installation

**⚠️ This is alpha research code. Install in development mode for exploration only.**

## Requirements

- Python 3.9-3.11
- NumPy, SciPy, Matplotlib
- HDF5 (for data I/O)

## Quick Install

```bash
git clone https://github.com/hmbown/analog-hawking-radiation.git
cd analog-hawking-radiation
pip install -e .
```

## Verify Installation

```bash
python -c "import analog_hawking; print('Installation successful')"
```

## Development Install

For development and exploration:

```bash
pip install -e ".[dev]"
pytest tests/test_eli_compatibility_system.py -v  # Run core tests
```

## Troubleshooting

- **Import errors**: Ensure src/ is in PYTHONPATH
- **Missing dependencies**: Install individually with pip
- **Test failures**: Expected for alpha code, see ALPHA_STATUS.md