# Contributing to Analog Hawking Radiation Simulation

Thank you for your interest in contributing to this research project. This repository contains a computational framework for simulating analog Hawking radiation in laser-plasma systems.

## AI Tool Usage Policy for Contributors

**Allowed:**
- AI code completion (Copilot, etc.) for boilerplate and tests
- AI documentation assistant (Claude) for drafting
- Human must review ALL AI-generated content before commit

**NOT Allowed:**
- AI commits under bot names (use your own GitHub account)
- AI-generated commits without human review
- Claiming AI tools as co-authors

**Commit Message Convention:**
If AI significantly assisted a commit, add "[AI-assisted]" to the message:

```bash
git commit -m "Add graybody model tests [AI-assisted]"
```

This maintains transparency without polluting the contributor graph.

**Why This Policy?**
We believe AI are tools, not collaborators. But we also believe in transparency.
This policy balances both values.
See HONESTY.md for the project's full AI disclosure statement.

## Scientific Standards

All contributions should maintain rigorous scientific standards:

- Implement physics models with proper validation against analytical solutions
- Include appropriate error handling and uncertainty quantification
- Follow established computational physics best practices
- Maintain reproducible and well-documented code
- Use appropriate statistical methods for data analysis

## Code Quality

- Follow PEP 8 style guidelines for Python code
- Include comprehensive unit tests for new functionality
- Document physics models with references to relevant literature
- Use clear variable names that reflect physical quantities
- Include appropriate comments explaining complex physics calculations

## Development Setup

- Install project with dev extras:
  ```bash
  pip install -e .[dev]
  ```

## Thresholds Provenance

Core physics breakdown thresholds are centralized in `configs/thresholds.yaml` (max |v|/c, max |dv/dx|, max intensity). Sweep scripts load these by default and also accept CLI overrides:

```bash
python scripts/sweep_gradient_catastrophe.py \
  --n-samples 500 \
  --output results/gradient_limits_production \
  --thresholds configs/thresholds.yaml \
  --vmax-frac 0.5 --dvdx-max 4e12 --intensity-max 6e50
```

## Keeping Docs in Sync

Do not hand-edit key numbers (κ_max, scaling exponents, 95% CIs, PIC κ/horizon) in `RESEARCH_HIGHLIGHTS.md`. Render docs from results JSONs instead:

```bash
python scripts/doc_sync/render_docs.py \
  --sweep results/gradient_limits_production/gradient_catastrophe_sweep.json \
  --pic results/pic_pipeline_summary.json
```

CI enforces that rendered docs match committed docs.

## Linting & Formatting

We use Ruff and Black.

```bash
ruff check .
black --check .
```

Use `black .` locally to format before committing.

## Documentation Standards

- Update documentation when adding new features or modifying existing functionality
- Use Markdown docs under `docs/` (no Sphinx). Keep sections concise and link to code when helpful
- Include mathematical notation where appropriate using LaTeX inline blocks in Markdown
- Provide small, runnable examples or reference `scripts/` entries
- Ensure all public APIs are properly documented with docstrings

## Pull Request Size & Tests

- Prefer focused PRs (≤ ~400 lines changed) to ease review
- Include unit tests for new functionality and update existing tests as needed
- Run the full test suite locally before opening a PR: `pytest -q`

## Pull Request Process

1. Fork the repository and create your branch from `main`
2. Add tests for any new functionality
3. Ensure all tests pass
4. Update documentation as appropriate
5. Submit a pull request with a clear description of the changes and their scientific justification

## Research Ethics

This project aims to advance understanding of analog gravity systems through rigorous computational modeling. All contributions should:

- Present results with appropriate statistical analysis
- Acknowledge limitations of the models
- Cite relevant literature appropriately
- Avoid overstating the implications of results
- Maintain scientific objectivity in all documentation

## Code of Conduct

### Our Pledge

In the interest of fostering an open and welcoming environment, we as contributors and maintainers pledge to making participation in our project and our community a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, gender identity and expression, level of experience, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards

Examples of behavior that contributes to creating a positive environment include:

- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

Examples of unacceptable behavior by participants include:

- The use of sexualized language or imagery and unwelcome sexual attention or advances
- Trolling, insulting/derogatory comments, and personal or political attacks
- Public or private harassment
- Publishing others' private information, such as a physical or electronic address, without explicit permission
- Other conduct which could reasonably be considered inappropriate in a professional setting

### Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable behavior and are expected to take appropriate and fair corrective action in response to any instances of unacceptable behavior.

Project maintainers have the right and responsibility to remove, edit, or reject comments, commits, code, wiki edits, issues, and other contributions that are not aligned to this Code of Conduct, or to ban temporarily or permanently any contributor for other behaviors that they deem inappropriate, threatening, offensive, or harmful.

### Scope

This Code of Conduct applies both within project spaces and in public spaces when an individual is representing the project or its community. Examples of representing a project or community include using an official project e-mail address, posting via an official social media account, or acting as an appointed representative at an online or offline event.

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project team. All complaints will be reviewed and investigated and will result in a response that is deemed necessary and appropriate to the circumstances. The project team is obligated to maintain confidentiality with regard to the reporter of an incident. Further details of specific enforcement policies may be posted separately.

Project maintainers who do not follow or enforce the Code of Conduct in good faith may face temporary or permanent repercussions as determined by other members of the project's leadership.

## Questions?

If you have questions about contributing to this research project, please open an issue for discussion.
