# Analysis Scripts

This directory contains Python scripts for analyzing simulation results, generating reports, and performing statistical analyses on the analog Hawking radiation dataset.

## Available Scripts

### Comprehensive Analysis

- **`comprehensive_analysis.py`** - Comprehensive analysis of hybrid_sweep.csv containing fluid vs hybrid model comparisons
  ```bash
  python scripts/analysis/comprehensive_analysis.py
  ```

- **`enhanced_comprehensive_analysis.py`** - Enhanced analysis for expanded datasets (≥100 configurations) with advanced uncertainty quantification
  ```bash
  python scripts/analysis/enhanced_comprehensive_analysis.py
  ```

### Dataset Comparison

- **`dataset_comparison_analysis.py`** - Compare original (20 configs) vs enhanced (≥100 configs) datasets
  ```bash
  python scripts/analysis/dataset_comparison_analysis.py
  ```

### Optimization Analysis

- **`optimization_analysis.py`** - Multi-objective optimization and Pareto frontier analysis
  ```bash
  python scripts/analysis/optimization_analysis.py
  ```

### Physics Integration Tests

- **`test_enhanced_physics_integration.py`** - Integration tests for enhanced physics models
  ```bash
  pytest scripts/analysis/test_enhanced_physics_integration.py
  ```

## Usage Guidelines

### Running Analysis Scripts

1. **Ensure data availability**: Most scripts expect data files in `results/` or current directory
2. **Check dependencies**: Some scripts may require additional packages (seaborn, scikit-learn, etc.)
3. **Review output**: Scripts typically generate figures and save results to `results/` subdirectories

### Common Workflow

```bash
# 1. Generate simulation data
make comprehensive

# 2. Run analysis scripts
python scripts/analysis/comprehensive_analysis.py
python scripts/analysis/optimization_analysis.py

# 3. Review results
ls results/
```

## Output Locations

- **Figures**: Typically saved to `results/figures/` or `docs/img/`
- **Data files**: CSV/JSON summaries saved to `results/`
- **Reports**: Markdown reports generated in `docs/reports/`

## Development

### Adding New Analysis Scripts

When creating new analysis scripts:

1. Use the shebang `#!/usr/bin/env python3`
2. Include a clear docstring with purpose, author, and date
3. Follow the existing naming convention: `{analysis_type}_analysis.py`
4. Add entry to this README
5. Include command-line argument parsing for flexibility
6. Add unit tests if the script contains reusable functions

### Code Style

- Follow PEP 8 style guidelines
- Use descriptive variable names
- Add comments for complex calculations
- Include warnings filters for known deprecations
- Set matplotlib style for consistent plotting

## Related Resources

- **[Main Scripts](../)** - Core simulation and workflow scripts
- **[Analysis Reports](../../docs/reports/)** - Generated analysis reports and summaries
- **[Test Suite](../../tests/)** - Unit and integration tests
- **[Documentation](../../docs/)** - User guides and methodology

## Dependencies

Common dependencies across analysis scripts:
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn
- scikit-optimize (for optimization analysis)

Install all dependencies:
```bash
pip install -r requirements.txt
pip install scikit-optimize seaborn  # Optional but recommended
```

## Troubleshooting

**Issue**: Script can't find data files
- **Solution**: Check that you've run the appropriate simulation scripts first, or specify data paths explicitly

**Issue**: Missing dependencies
- **Solution**: `pip install <package>` or use `pip install -e .[dev]` for all development dependencies

**Issue**: Matplotlib backend errors (headless systems)
- **Solution**: Set `MPLBACKEND=Agg` environment variable or use `--no-plots` flag if available

For more help, see the main [FAQ](../../docs/FAQ.md) or open an issue.
