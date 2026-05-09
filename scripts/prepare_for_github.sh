#!/usr/bin/env bash
# Pre-push helper for ChromaGraphNet
#
# Run this once before your first `git push` to:
#   1. Replace the USERNAME placeholder in metadata files with your real
#      GitHub handle.
#   2. Verify all unit tests pass.
#   3. Confirm the working tree is clean and ready to push.
#
# Usage:
#   ./scripts/prepare_for_github.sh <your-github-username>
#
# Example:
#   ./scripts/prepare_for_github.sh kroy3

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <your-github-username>"
    echo "Example: $0 kroy3"
    exit 1
fi

GH_USER="$1"

echo "==> Replacing 'USERNAME' with '$GH_USER' in metadata files..."
# macOS sed and GNU sed differ; this works on both via Python
python - <<EOF
import os, re
files = [
    'pyproject.toml',
    'README.md',
    'AUTHORS.md',
    'MODEL_CARD.md',
    'CONTRIBUTING.md',
    'CITATION.cff',
]
for f in files:
    if not os.path.exists(f):
        continue
    text = open(f).read()
    new = text.replace('USERNAME/chromagraphnet', '$GH_USER/chromagraphnet')
    if new != text:
        open(f, 'w').write(new)
        print(f"  Updated {f}")
EOF

echo
echo "==> Running test suite..."
python -m pytest tests/ -q || {
    echo "ERROR: tests failed; do not push until tests pass."
    exit 1
}

echo
echo "==> Verifying CLI..."
chromagraphnet-info > /dev/null && echo "  chromagraphnet-info OK"

echo
echo "==> Final repository status:"
git status
echo
echo "==> Done!  You're ready to:"
echo "  git add -u"
echo "  git commit -m 'docs: replace USERNAME placeholder with $GH_USER'"
echo "  git remote add origin git@github.com:$GH_USER/chromagraphnet.git"
echo "  git push -u origin main"
echo "  git push origin v0.1.1"
