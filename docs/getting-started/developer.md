# Getting Started: Developer

**Quick Navigation**: [Back to Overview](../index.md) | [Quick Links](../QUICKLINKS.md) | [Full Documentation](../index.md)

This guide is for **developers** who want to:
- Contribute code and features
- Fix bugs and improve performance
- Add new physics models
- Enhance documentation and tooling

---

## 🛠️ Development Setup (5 minutes)

### 1. Fork and Clone
```bash
# Fork on GitHub first, then:
git clone https://github.com/YOUR_USERNAME/analog-hawking-radiation.git
cd analog-hawking-radiation
```

### 2. Automated Setup
```bash
# Sets up development environment, installs dependencies, runs tests
ahr dev --setup
```

### 3. Verify Installation
```bash
# Check everything works
ahr info
ahr validate --dashboard
ahr quickstart
```

### 4. Install Pre-commit Hooks
```bash
pre-commit install
```

---

## 📁 Repository Structure

```
analog-hawking-radiation/
├── src/analog_hawking/          # Core package
│   ├── __init__.py
│   ├── cli/                     # Command-line interface
│   │   └── main.py             # ahr command implementation
│   ├── physics_engine/          # Core physics algorithms
│   │   ├── horizon.py          # Horizon finding
│   │   ├── graybody_nd.py      # Graybody models
│   │   └── enhanced_*.py       # Experimental features
│   ├── detection/               # Detection modeling
│   ├── pipelines/               # Analysis pipelines
│   ├── utils/                   # Utilities
│   └── config/                  # Configuration
├── scripts/                     # Analysis scripts
├── tests/                       # Test suite
├── docs/                        # Documentation
├── examples/                    # Example code
├── notebooks/                   # Jupyter notebooks
├── configs/                     # Configuration files
└── results/                     # Output directory
```

---

## 🧪 Testing Strategy

### Test Structure

```bash
tests/
├── test_horizon.py              # Horizon finding tests
├── test_graybody.py             # Graybody model tests
├── test_detection.py            # Detection modeling tests
├── test_validation.py           # Validation framework tests
├── test_cli.py                  # CLI command tests
└── integration/                 # Integration tests
```

### Running Tests

```bash
# Quick test
make test                  # or: pytest -q

# Full test suite with coverage
pytest -v --cov=src/analog_hawking

# Specific test file
pytest tests/test_horizon.py -v

# Specific test
pytest tests/test_horizon.py::test_find_horizons -v

# With GPU tests
pytest -m gpu -v
```

### Writing New Tests

```python
# tests/test_my_new_feature.py
import pytest
import numpy as np
from analog_hawking.physics_engine import my_new_feature

def test_my_feature_basic():
    """Test basic functionality"""
    x = np.linspace(0, 100e-6, 1000)
    result = my_new_feature(x, param=1.0)
    
    assert result is not None
    assert len(result) == len(x)
    assert np.all(np.isfinite(result))

def test_my_feature_analytical():
    """Test against known analytical solution"""
    # Setup known case
    x = np.array([0, 50e-6, 100e-6])
    expected = np.array([0, 1, 0])
    
    result = my_new_feature(x, param=1.0)
    
    np.testing.assert_allclose(result, expected, rtol=1e-10)

def test_my_feature_edge_cases():
    """Test edge cases and error handling"""
    # Empty input
    with pytest.raises(ValueError):
        my_new_feature(np.array([]))
    
    # Invalid parameters
    with pytest.raises(ValueError):
        my_new_feature(np.array([1, 2, 3]), param=-1.0)
```

---

## 🏗️ Architecture & Design

### Core Design Principles

1. **Modularity**: Separate physics, numerics, and I/O
2. **Validation**: Every physics feature has tests
3. **Performance**: NumPy/CuPy for array operations
4. **Reproducibility**: Manifests track provenance
5. **Clarity**: Type hints and docstrings throughout

### Key Components

#### Horizon Finding (`physics_engine/horizon.py`)

```python
# Key functions
def find_horizons_with_uncertainty(x, v, cs, method='acoustic_exact'):
    """Find sonic horizons with uncertainty estimates"""
    # ... implementation
    
def compute_kappa_acoustic_exact(x, v, cs, horizon_idx):
    """Acoustic exact κ calculation"""
    # ... implementation

def compute_kappa_geometric(x, v, cs, horizon_idx):
    """Geometric κ approximation"""
    # ... implementation
```

#### Graybody Models (`physics_engine/graybody_nd.py`)

```python
def compute_graybody_transmission(omega, kappa, method='acoustic_wkb'):
    """Compute frequency-dependent transmission"""
    # ... implementation

def compute_tortoise_coordinate(x, v, cs):
    """Transform to tortoise coordinate"""
    # ... implementation
```

#### CLI Interface (`cli/main.py`)

```python
def cmd_quickstart(args: argparse.Namespace) -> int:
    """Quickstart command implementation"""
    # ... implementation
    return 0  # Success

def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser"""
    # ... implementation
```

---

## 🔧 Adding New Features

### Contribution Workflow

```bash
# 1. Create feature branch
git checkout -b feature/my-new-feature

# 2. Make changes (see examples below)
# ... edit files ...

# 3. Add tests
echo "def test_my_feature(): ..." >> tests/test_my_feature.py

# 4. Run tests and linting
pytest tests/test_my_feature.py -v
pre-commit run --all-files

# 5. Update documentation
# ... edit docs/ ...

# 6. Commit and push
git add .
git commit -m "feat: add my new feature with tests"
git push origin feature/my-new-feature

# 7. Create pull request on GitHub
```

### Example: Adding a New κ Method

```python
# src/analog_hawking/physics_engine/horizon.py

def compute_kappa_my_method(x, v, cs, horizon_idx):
    """
    My new κ calculation method.
    
    Parameters
    ----------
    x : array
        Position grid [m]
    v : array  
        Flow velocity [m/s]
    cs : array
        Sound speed [m/s]
    horizon_idx : int
        Index of horizon crossing
        
    Returns
    -------
    kappa : float
        Surface gravity [s⁻¹]
    uncertainty : float
        Numerical uncertainty [s⁻¹]
    """
    # Your implementation
    # ...
    return kappa, uncertainty

# Register the method
KAPPA_METHODS['my_method'] = compute_kappa_my_method
```

### Example: New CLI Command

```python
# src/analog_hawking/cli/main.py

def cmd_my_analysis(args: argparse.Namespace) -> int:
    """
    Run my custom analysis.
    
    Examples
    --------
    ahr my-analysis --input data.npy --output results/
    """
    import numpy as np
    from pathlib import Path
    
    # Load input
    data = np.load(args.input)
    
    # Your analysis
    result = my_analysis_function(data, param=args.param)
    
    # Save output
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'result.npy', result)
    
    print(f"Analysis complete. Results saved to {output_dir}")
    return 0

# Add to parser
def build_parser() -> argparse.ArgumentParser:
    # ... existing code ...
    
    my_cmd = sub.add_parser('my-analysis', help='Run my custom analysis')
    my_cmd.add_argument('--input', type=str, required=True, help='Input data file')
    my_cmd.add_argument('--output', type=str, default='results/my_analysis', help='Output directory')
    my_cmd.add_argument('--param', type=float, default=1.0, help='Analysis parameter')
    my_cmd.set_defaults(func=cmd_my_analysis)
    
    return p
```

---

## 🎨 Code Style & Quality

### Linting and Formatting

```bash
# Run all checks
make lint                  # or: pre-commit run --all-files

# Individual tools
ruff check src/ tests/ scripts/
black src/ tests/ scripts/

# Type checking
mypy src/analog_hawking/
```

### Type Hints

```python
from typing import Union, Optional, Tuple, Dict, Any
import numpy as np

def my_function(
    x: np.ndarray,
    param: float,
    method: str = 'default',
    tolerance: Optional[float] = None
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Function with type hints.
    
    Parameters
    ----------
    x : np.ndarray
        Input array
    param : float
        Control parameter
    method : str
        Calculation method
    tolerance : float, optional
        Numerical tolerance
        
    Returns
    -------
    result : np.ndarray
        Computed result
    metadata : dict
        Additional information
    """
    # Implementation
    pass
```

### Docstring Standards

Use NumPy/SciPy style docstrings:

```python
def compute_horizon_properties(
    x: np.ndarray,
    v: np.ndarray,
    cs: np.ndarray,
    **kwargs
) -> HorizonResult:
    """
    Compute properties of sonic horizons in plasma flow.
    
    Parameters
    ----------
    x : np.ndarray
        Position grid [m]
    v : np.ndarray
        Flow velocity [m/s]
    cs : np.ndarray
        Sound speed [m/s]
    method : str, optional
        Horizon finding method ('acoustic_exact', 'geometric')
        
    Returns
    -------
    HorizonResult
        Object containing horizon positions, kappa values, and uncertainties
        
    Notes
    -----
    The horizon is defined where |v| = c_s. Multiple methods for computing
    surface gravity κ are available, see [1]_.
    
    References
    ----------
    .. [1] Visser, M. "Acoustic black holes", 1998
    
    Examples
    --------
    >>> x = np.linspace(0, 100e-6, 1000)
    >>> v = 2e6 * np.tanh((x - 50e-6) / 10e-6)
    >>> cs = np.full_like(x, 1e6)
    >>> result = compute_horizon_properties(x, v, cs)
    >>> print(f"Found {result.n_horizons} horizons")
    Found 2 horizons
    """
    # Implementation
    pass
```

---

## 🐛 Debugging

### Common Issues

1. **Import Errors**
```bash
# Fix: Reinstall in editable mode
pip install -e .
```

2. **Test Failures**
```bash
# Run specific test with verbose output
pytest tests/test_horizon.py::test_find_horizons -v -s

# Check if golden files exist
ls goldens/
```

3. **Performance Issues**
```bash
# Profile code
python -m cProfile -o profile.stats scripts/my_script.py
snakeviz profile.stats

# Memory profiling
mprof run scripts/my_script.py
mprof plot
```

4. **GPU Issues**
```bash
# Check GPU availability
ahr gpu-info

# Force CPU for debugging
export ANALOG_HAWKING_FORCE_CPU=1
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Or use environment variable
export ANALOG_HAWKING_LOG_LEVEL=DEBUG
```

---

## 📊 Performance Optimization

### Profiling

```bash
# Time a specific command
time ahr quickstart

# Python profiling
python -m cProfile -o quickstart.prof -m analog_hawking.cli.main quickstart
pyprof2calltree -i quickstart.prof -o quickstart.callgrind
# Open with QCacheGrind or similar
```

### GPU Acceleration

```python
# Check if CuPy is available
try:
    import cupy as cp
    USE_GPU = True
except ImportError:
    USE_GPU = False

# Use appropriate array module
from analog_hawking.utils import array_module as am

# Now use am (either numpy or cupy)
x = am.linspace(0, 100, 1000)
y = am.sin(x)
```

### Optimization Tips

1. **Vectorization**: Use NumPy operations, not loops
2. **Memory**: Pre-allocate arrays, avoid copies
3. **GPU**: Move data to GPU once, keep it there
4. **I/O**: Batch file operations, use efficient formats (NPZ, HDF5)

---

## 🚀 Advanced Development

### Creating New Physics Modules

```python
# src/analog_hawking/physics_engine/my_physics.py
"""
My new physics module.

This module implements [brief description].
"""

from typing import Tuple
import numpy as np
from ..utils.array_module import array_module as am

__all__ = ['my_physics_function']

def my_physics_function(
    x: np.ndarray,
    param: float,
    method: str = 'default'
) -> Tuple[np.ndarray, dict]:
    """
    Compute my physics quantity.
    
    Detailed description here...
    """
    # Implementation
    pass
```

### Adding Configuration Options

```python
# src/analog_hawking/config/schemas.py
from pydantic import BaseModel, Field
from typing import Optional

class MyConfig(BaseModel):
    """Configuration for my feature."""
    
    param: float = Field(
        default=1.0,
        description="Control parameter",
        gt=0,
        le=100
    )
    
    method: str = Field(
        default='default',
        description="Calculation method",
        choices=['default', 'fast', 'accurate']
    )
    
    tolerance: Optional[float] = Field(
        default=None,
        description="Numerical tolerance",
        gt=0
    )
```

---

## 📅 Release Process

### Version Bumping

```bash
# Update version in __init__.py
# Update CHANGELOG.md
# Create git tag
git tag -a v0.4.0 -m "Release version 0.4.0"
git push origin v0.4.0
```

### Pre-release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped
- [ ] New features validated
- [ ] Performance benchmarks run
- [ ] Tutorial notebooks tested

---

## 🤝 Contributing Guidelines

### Pull Request Process

1. **Small PRs**: Keep changes focused (< 400 lines)
2. **Tests**: Include tests for new functionality
3. **Docs**: Update documentation
4. **Validation**: Run full test suite
5. **Description**: Clear PR description with motivation

### Code Review

- Be respectful and constructive
- Focus on physics correctness
- Check test coverage
- Verify documentation
- Performance implications

### Commit Messages

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code restructuring
- `perf:` Performance improvements

Examples:
```
feat: add new κ calculation method
docs: improve horizon finding tutorial
test: add convergence tests for graybody models
```

---

## 📞 Getting Help

**Development questions?**
- [GitHub Discussions](https://github.com/hmbown/analog-hawking-radiation/discussions)
- Tag: "development"

**Code review needed?**
- Open draft PR
- Ask specific questions in PR description

**Architecture decisions?**
- Email: hunter@shannonlabs.dev
- Subject: "Analog Hawking - Architecture"

---

## 🎯 Development Roadmap

**Near-term** (next release):
- [ ] Enhanced CLI with more commands
- [ ] Improved documentation structure
- [ ] Better error handling and messages
- [ ] Performance optimizations

**Medium-term** (2-3 releases):
- [ ] 2D/3D horizon finding
- [ ] GPU acceleration improvements
- [ ] Real-time visualization
- [ ] Experimental data ingestion

**Long-term** (future):
- [ ] Full PIC integration
- [ ] Multi-physics coupling
- [ ] Interactive web interface
- [ ] Cloud execution support

See [phase timeline](../phase_timeline.md) for detailed roadmap.

---

<div align="center">

**[Back to Overview](../index.md)** | **[Quick Links](../QUICKLINKS.md)** | **[Full Documentation](../index.md)**

*Laboratory Black Hole Detection, Quantified*

</div>
