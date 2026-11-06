#!/usr/bin/env bash
# Train one atom with the measured configuration (upstream ferminet entry point).
# Usage: ATOM=Xe [SPIN=0] [BATCH=1024] [ITERATIONS=...] bash scripts/run_atom.sh
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
ATOM=${ATOM:?set ATOM, e.g. Fe Kr Cd Xe}
case "$ATOM" in Fe) DEF_SPIN=4 ;; *) DEF_SPIN=0 ;; esac
source "$ROOT/scripts/configs/faster.env"
cd "$ROOT"
JAX_PLATFORMS=${JAX_PLATFORMS:-cuda} "$PYTHON" -m ferminet.main \
  --config ferminet/configs/atom.py \
  --config.system.atom="$ATOM" \
  --config.system.spin_polarisation="${SPIN:-$DEF_SPIN}" \
  --config.batch_size="${BATCH:-1024}" \
  --config.pretrain.basis=ano-rcc \
  ${ITERATIONS:+--config.optim.iterations=$ITERATIONS} \
  ${SAVE_PATH:+--config.log.save_path=$SAVE_PATH} "$@"
