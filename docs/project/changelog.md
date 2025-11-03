# Changelog

The full changelog is maintained in the repository root.
[View CHANGELOG.md on GitHub](https://github.com/Hmbown/analog-hawking-radiation/blob/main/CHANGELOG.md).

## 2025-11-03

- Regenerated gradient-catastrophe highlights (`python scripts/doc_sync/render_docs.py --sweep results/gradient_limits_production/gradient_catastrophe_sweep.json --pic results/pic_pipeline_summary.json`) and committed outputs.
- Notebook freshness: execute `python -m nbconvert --execute notebooks/01_quickstart.ipynb --to notebook --output notebooks/01_quickstart.executed.ipynb` (repeat for other notebooks) prior to external releases; nbval smoke target under consideration.
- Physics validation now reports 58/58 passing after causality tolerance and ADK log-rate fixes (`ahr validate`).
