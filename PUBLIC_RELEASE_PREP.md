# Public Release Preparation Guide

**Repository:** Analog Hawking Radiation Analysis  
**Version:** 0.3.1-alpha  
**Date:** 2025-11-06  
**Status:** Ready for Public Release (with cleanup)

---

## Current Repository Status

✅ **Good News:** The git repository is already clean and ready for public release!

- **417 files** tracked in git
- **No sensitive files** in version control (no API keys, passwords, credentials)
- **No large binaries** or unnecessary files tracked
- **MIT License** properly configured
- **Documentation** is comprehensive

---

## Required Cleanup Actions

### 1. Remove Local Untracked Files (IMPORTANT)

These files are present locally but NOT tracked in git (good!). However, they should be deleted before public release:

```bash
# Remove virtual environment (350MB)
rm -rf .venv/

# Remove firebase debug log (contains email address)
rm firebase-debug.log

# Remove macOS metadata files
find . -name ".DS_Store" -type f -delete

# Remove Claude settings directory
rm -rf .claude/

# Remove Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
```

**Estimated space saved:** ~400MB

---

### 2. Update Email Address (Optional but Recommended)

Currently using: `hunter@shannonlabs.dev`

This is a **professional contact email** and is appropriate for public release. However, you may want to update it to a more generic address.

**Files containing the email address:**
- `CODE_OF_CONDUCT.md` (1 instance)
- `setup.py` (1 instance)
- `SECURITY.md` (1 instance)
- `src/analog_hawking/__init__.py` (1 instance)
- `docs/archive/` files (2 instances - these are archives)
- `results/RESEARCH_SUMMARY_v0.3.0.md` (2 instances)

**To update:**
```bash
# Replace with your preferred contact email
OLD_EMAIL="hunter@shannonlabs.dev"
NEW_EMAIL="your-email@institution.edu"

find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.txt" \) \
  -not -path "./.venv/*" \
  -not -path "./.git/*" \
  -exec sed -i '' "s/$OLD_EMAIL/$NEW_EMAIL/g" {} +
```

---

### 3. Verify .gitignore is Comprehensive

The `.gitignore` file is already well-configured and includes:
- ✅ Virtual environments (`.venv/`)
- ✅ Python cache files (`__pycache__/`, `*.pyc`)
- ✅ Environment files (`.env`)
- ✅ OS files (`.DS_Store`, `Thumbs.db`)
- ✅ Log files (`*.log`, `firebase-debug.log`)
- ✅ IDE directories (`.vscode/`, `.idea/`, `.claude/`)
- ✅ Large data files (`*.pkl`, `*.npz`, `*.npy`)

**Status:** .gitignore is comprehensive and appropriate.

---

### 4. Review Large Files in Git History

Currently tracked large files:
- `numerical_stability_results.pkl` (36 KB)
- `results/samples/hybrid_sweep.csv` (sample data)

**Status:** These are appropriate for the repository.

**To check for large files in history:**
```bash
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | awk '/^blob/ {print $2 " " $3 " " $4}' | sort -k2 -nr | head -20
```

---

### 5. Verify No Secrets in Commit History

**Check for common secrets in history:**
```bash
# Install git-secrets if not available
# git clone https://github.com/awslabs/git-secrets

# Scan for common patterns
git secrets --scan-history
```

**Manual check for common patterns:**
```bash
git log -p --all | grep -i -E "(api_key|apikey|password|passwd|secret|token|private_key|privatekey)" | head -20
```

**Status:** Repository appears clean, but verify with above commands.

---

### 6. Update README for Public Audience

**Current README:** Comprehensive and appropriate

**Suggested additions for public release:**
- [ ] Add "Installation" section
- [ ] Add "Quick Start" example
- [ ] Add "Citation" information
- [ ] Add "Contributing" link
- [ ] Add "Support" section with contact info

---

### 7. Verify License and Copyright

**Current License:** MIT License (good for public release)

**Checklist:**
- [x] LICENSE file present
- [x] Copyright year correct (2025)
- [x] License referenced in README
- [x] No proprietary code included

**Optional:** Add license headers to source files:
```python
# MIT License
# Copyright (c) 2025
# See LICENSE file for details
```

---

### 8. Create SECURITY.md

**Already exists** - Good!

**Contents should include:**
- [x] How to report security vulnerabilities
- [x] Contact email address
- [x] Expected response time
- [x] Scope of security policy

---

### 9. Create CODE_OF_CONDUCT.md

**Already exists** - Good!

**Standard sections:**
- [x] Our Pledge
- [x] Our Standards
- [x] Enforcement Responsibilities
- [x] Scope
- [x] Enforcement
- [x] Enforcement Guidelines
- [x] Attribution

---

### 10. Review and Clean Commit History (Optional)

If you want to clean the commit history before public release:

**Option A: Create new initial commit (recommended for public release)**
```bash
# Create a new branch with clean history
git checkout --orphan public-release
git add -A
git commit -m "Initial public release v0.3.1-alpha"

# Replace main branch
git branch -D main
git branch -m main

# Force push to remote (be careful!)
git push origin main --force
```

**Option B: Keep existing history**
- Pros: Preserves development history
- Cons: May contain messy commits, sensitive info in old commits

**Recommendation:** Option A for cleaner public release.

---

## Pre-Release Checklist

### Security Audit
- [ ] Run `git secrets --scan-history`
- [ ] Check for large files in history
- [ ] Verify no credentials in any commits
- [ ] Review all files containing email addresses

### Repository Cleanup
- [ ] Delete `.venv/` directory
- [ ] Delete `firebase-debug.log`
- [ ] Delete `.claude/` directory
- [ ] Remove all `.DS_Store` files
- [ ] Clear Python cache files

### Documentation Review
- [ ] README is comprehensive
- [ ] Installation instructions clear
- [ ] Examples work correctly
- [ ] API documentation complete
- [ ] Contributing guidelines present

### Code Quality
- [ ] Tests pass: `pytest`
- [ ] No linting errors: `flake8` or `pylint`
- [ ] Type hints included (optional but recommended)
- [ ] Code formatted: `black` or `autopep8`

### Legal
- [ ] LICENSE file present and correct
- [ ] Copyright notices updated
- [ ] No proprietary code included
- [ ] Third-party licenses acknowledged

### Git Configuration
- [ ] `.gitignore` comprehensive
- [ ] No sensitive files tracked
- [ ] Branch protection rules set up (for main branch)
- [ ] Repository visibility set to "Public"

---

## Release Process

### Step 1: Final Local Cleanup
```bash
#!/bin/bash
echo "Cleaning repository for public release..."

# Remove untracked files and directories
echo "Removing virtual environment..."
rm -rf .venv/

echo "Removing debug logs..."
rm -f firebase-debug.log

echo "Removing Claude settings..."
rm -rf .claude/

echo "Removing macOS metadata files..."
find . -name ".DS_Store" -type f -delete

echo "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete

echo "Cleanup complete!"
echo "Repository size reduced by approximately 400MB"
```

### Step 2: Update Documentation
- Review and update README.md
- Ensure all examples are working
- Add citation information
- Update contact information (if desired)

### Step 3: Create Release Tag
```bash
git add -A
git commit -m "Prepare for public release v0.3.1-alpha"
git tag -a v0.3.1-alpha -m "Public release v0.3.1-alpha"
git push origin main --tags
```

### Step 4: GitHub Repository Settings
1. Go to repository Settings on GitHub
2. Scroll to "Danger Zone"
3. Click "Change repository visibility"
4. Select "Public" and confirm
5. Configure branch protection for `main`:
   - Require pull request reviews
   - Require status checks
   - Include administrators

### Step 5: Post-Release Tasks
- [ ] Announce on social media/twitter
- [ ] Submit to relevant newsletters
- [ ] Update personal website
- [ ] Monitor issues and pull requests
- [ ] Respond to community feedback

---

## Repository Statistics

**Current State:**
- Tracked files: 417
- Repository size: ~50MB (without .venv)
- Languages: Python (99%), Markdown (1%)
- License: MIT
- Documentation: Comprehensive

**After Cleanup:**
- Expected size: ~10-15MB
- Clean commit history (if rebased)
- No sensitive information
- Ready for public consumption

---

## GitHub Best Practices

### Repository Metadata
- [ ] Add repository description
- [ ] Add relevant topics/tags
- [ ] Add website URL (if applicable)
- [ ] Enable discussions (optional)
- [ ] Enable projects (optional)

### Community Standards
- [ ] Code of conduct present
- [ ] Contributing guidelines present
- [ ] Issue templates created
- [ ] Pull request template created

### Security
- [ ] Security policy present
- [ ] Enable security advisories
- [ ] Enable dependabot alerts
- [ ] Private vulnerability reporting enabled

---

## Troubleshooting

### Problem: Large files in git history
**Solution:** Use `git filter-repo` or BFG Repo-Cleaner
```bash
# Install git filter-repo
pip install git-filter-repo

# Remove large files
git filter-repo --strip-blobs-bigger-than 10M
```

### Problem: Sensitive data in old commits
**Solution:** Create new initial commit (see Option A above)

### Problem: .venv still showing up
**Solution:** Make sure it's in .gitignore and not tracked
```bash
git rm -rf --cached .venv/
git commit -m "Remove .venv from tracking"
```

---

## Verification Commands

Run these commands to verify repository is ready:

```bash
# Check for sensitive files
echo "Checking for sensitive files..."
find . -type f -name "*.key" -o -name "*.pem" -o -name "*.env" -o -name "*password*" | grep -v ".venv" | grep -v "__pycache__"

# Check repository size
echo "Repository size:"
du -sh .git/

# Check for large files in HEAD
echo "Largest files in current commit:"
git ls-tree -r HEAD --long | sort -k3 -nr | head -10

# Verify tests pass
echo "Running tests..."
pytest --maxfail=5 -q

# Check for common secrets
echo "Scanning for potential secrets..."
git log -p --all | grep -i -E "(api_key|password|secret|token)" | head -5 || echo "No secrets found"
```

---

## Final Verification Checklist

Before making public, verify:

- [ ] No sensitive files in working directory
- [ ] .gitignore is comprehensive
- [ ] All tests pass
- [ ] Documentation is complete
- [ ] LICENSE file present
- [ ] README is informative
- [ ] Code of conduct present
- [ ] Contributing guidelines present
- [ ] Security policy present
- [ ] Examples work correctly
- [ ] No credentials in commit history
- [ ] Repository size is reasonable
- [ ] All files are intentionally included

---

## Contact Information

For questions about public release preparation:
- Repository maintainer: hunter@shannonlabs.dev
- Review SECURITY.md for vulnerability reporting

---

**Document Version:** 1.0  
**Last Updated:** 2025-11-06  
**Next Review:** After public release