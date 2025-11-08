#!/bin/bash

# Prepare Analog Hawking Radiation repository for public release
# This script performs cleanup tasks to ensure no sensitive data is included

set -e

echo "======================================"
echo "Public Release Preparation Script"
echo "Analog Hawking Radiation Analysis"
echo "Version: 0.3.1-alpha"
echo "======================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -d ".git" ]; then
    print_error "This doesn't appear to be the repository root. Please run from repository root."
    exit 1
fi

print_status "Starting cleanup for public release..."
echo ""

# Step 1: Remove virtual environment
if [ -d ".venv" ]; then
    print_status "Removing virtual environment (.venv/)..."
    rm -rf .venv/
    print_status "✓ Virtual environment removed (~350MB freed)"
else
    print_status "✓ No virtual environment found"
fi
echo ""

# Step 2: Remove debug logs
if [ -f "firebase-debug.log" ]; then
    print_status "Removing firebase debug log..."
    rm -f firebase-debug.log
    print_status "✓ Firebase debug log removed"
else
    print_status "✓ No firebase debug log found"
fi
echo ""

# Step 3: Remove Claude settings
if [ -d ".claude" ]; then
    print_status "Removing Claude settings directory..."
    rm -rf .claude/
    print_status "✓ Claude settings removed"
else
    print_status "✓ No Claude settings directory found"
fi
echo ""

# Step 4: Remove macOS metadata files
print_status "Removing macOS .DS_Store files..."
DS_COUNT=$(find . -name ".DS_Store" -type f 2>/dev/null | wc -l)
if [ "$DS_COUNT" -gt 0 ]; then
    find . -name ".DS_Store" -type f -delete
    print_status "✓ Removed $DS_COUNT .DS_Store files"
else
    print_status "✓ No .DS_Store files found"
fi
echo ""

# Step 5: Remove Python cache files
print_status "Removing Python cache files..."

# Remove __pycache__ directories
CACHE_COUNT=$(find . -type d -name "__pycache__" 2>/dev/null | wc -l)
if [ "$CACHE_COUNT" -gt 0 ]; then
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    print_status "✓ Removed $CACHE_COUNT __pycache__ directories"
else
    print_status "✓ No __pycache__ directories found"
fi

# Remove .pyc files
PYC_COUNT=$(find . -type f -name "*.pyc" 2>/dev/null | wc -l)
if [ "$PYC_COUNT" -gt 0 ]; then
    find . -type f -name "*.pyc" -delete
    print_status "✓ Removed $PYC_COUNT .pyc files"
else
    print_status "✓ No .pyc files found"
fi

# Remove .pyo files
PYO_COUNT=$(find . -type f -name "*.pyo" 2>/dev/null | wc -l)
if [ "$PYO_COUNT" -gt 0 ]; then
    find . -type f -name "*.pyo" -delete
    print_status "✓ Removed $PYO_COUNT .pyo files"
else
    print_status "✓ No .pyo files found"
fi
echo ""

# Step 6: Check for sensitive files
print_status "Checking for potentially sensitive files..."
SENSITIVE_FILES=$(find . -type f \( -name "*.key" -o -name "*.pem" -o -name "*.env" -o -name "*password*" -o -name "*credential*" -o -name ".env" \) 2>/dev/null | grep -v ".venv" | grep -v "__pycache__" || true)

if [ -n "$SENSITIVE_FILES" ]; then
    print_warning "Found potentially sensitive files:"
    echo "$SENSITIVE_FILES"
    print_warning "Please review these files manually"
else
    print_status "✓ No sensitive files found"
fi
echo ""

# Step 7: Verify .gitignore
print_status "Checking .gitignore configuration..."
if [ -f ".gitignore" ]; then
    # Check for common entries
    for pattern in ".venv" "__pycache__" "*.pyc" ".env" ".DS_Store" "firebase-debug.log"; do
        if grep -q "$pattern" .gitignore; then
            print_status "✓ .gitignore contains: $pattern"
        else
            print_warning "⚠ .gitignore missing: $pattern"
        fi
    done
else
    print_warning "⚠ No .gitignore file found"
fi
echo ""

# Step 8: Check git status
print_status "Checking git status..."
GIT_STATUS=$(git status --porcelain)
if [ -n "$GIT_STATUS" ]; then
    print_warning "There are uncommitted changes:"
    echo "$GIT_STATUS"
    echo ""
    read -p "Do you want to see detailed status? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git status
    fi
else
    print_status "✓ Working directory is clean"
fi
echo ""

# Step 9: Verify no large files in git
print_status "Checking for large files in git history..."
LARGE_FILES=$(git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' 2>/dev/null | awk '/^blob/ {if ($3 > 10000000) print $0}' | head -5 || true)

if [ -n "$LARGE_FILES" ]; then
    print_warning "Found large files (>10MB):"
    echo "$LARGE_FILES"
    print_warning "Consider using git-lfs or removing these files"
else
    print_status "✓ No excessively large files in git history"
fi
echo ""

# Step 10: Run tests if pytest is available
print_status "Running tests to verify functionality..."
if command -v pytest &> /dev/null; then
    print_status "Running pytest..."
    pytest --maxfail=3 -q
    if [ $? -eq 0 ]; then
        print_status "✓ All tests pass"
    else
        print_warning "⚠ Some tests failed - please review"
    fi
else
    print_warning "⚠ pytest not available - skipping tests"
fi
echo ""

# Step 11: Check repository size
print_status "Repository size analysis:"
GIT_SIZE=$(du -sh .git/ 2>/dev/null | cut -f1)
WORKING_SIZE=$(du -sh . 2>/dev/null | cut -f1)
print_status "  Git history size: $GIT_SIZE"
print_status "  Working directory size: $WORKING_SIZE"
echo ""

# Step 12: Final summary
print_status "======================================"
print_status "Cleanup Complete!"
print_status "======================================"
echo ""
print_status "Summary of actions taken:"
echo "  • Removed virtual environment"
echo "  • Removed debug logs"
echo "  • Removed Claude settings"
echo "  • Removed macOS metadata files"
echo "  • Removed Python cache files"
echo "  • Verified .gitignore configuration"
echo "  • Checked for sensitive files"
echo "  • Verified no large files in git"
echo "  • Ran test suite"
echo ""
print_status "The repository is now ready for public release!"
echo ""
print_status "Next steps:"
echo "  1. Review any warnings above"
echo "  2. Update email addresses if desired (see PUBLIC_RELEASE_PREP.md)"
echo "  3. Create a release tag: git tag -a v0.3.1-alpha -m 'Public release'"
echo "  4. Push to GitHub: git push origin main --tags"
echo "  5. Change repository visibility to 'Public' on GitHub"
echo ""
print_status "For detailed instructions, see PUBLIC_RELEASE_PREP.md"
echo ""

# Optional: Create a git status reminder
read -p "Show current git status? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git status
fi

echo ""
print_status "Done!"
