#!/usr/bin/env bash
# Clone the original FermiNet at the base commit into third_party/ferminet-upstream
# (the reference tree for timing and bit-wise comparisons).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEST=${REFERENCE_TREE:-$ROOT/third_party/ferminet-upstream}
COMMIT=e9d5a604d15c617864e7caefd948d87453b01ae6
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/google-deepmind/ferminet.git "$DEST"
fi
git -C "$DEST" checkout --quiet "$COMMIT"
echo "reference tree: $DEST @ $(git -C "$DEST" rev-parse --short HEAD)"
