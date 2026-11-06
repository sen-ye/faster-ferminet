#!/usr/bin/env bash
# Bit-wise check only (no timing): reference vs Faster-FermiNet in deterministic mode.
# Env: GPUS, PYTHON, REFERENCE_TREE, OUT_DIR, CHECK_ITERS (default 8), LONG_ITERS (default 0).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
GPUS=${GPUS:-0,1,2,3}
REFERENCE_TREE=${REFERENCE_TREE:-$ROOT/third_party/ferminet-upstream}
OUT_DIR=${OUT_DIR:-$ROOT/runs/bitwise-$(date +%Y%m%d-%H%M%S)}
source "$ROOT/scripts/configs/faster.env"
"$PYTHON" "$ROOT/bench/make_jobs.py" --plan bitwise --reference "$REFERENCE_TREE" --candidate "$ROOT" \
  --out "$OUT_DIR" --gpus "$GPUS" --check-iters "${CHECK_ITERS:-8}" --long-iters "${LONG_ITERS:-0}"
PYTHON="$PYTHON" "$PYTHON" "$ROOT/bench/launch.py" "$OUT_DIR/jobs.json" "$GPUS"
"$PYTHON" -I "$ROOT/bench/compare.py" "$OUT_DIR/ref_det" "$OUT_DIR/cand_det"
if [ "${LONG_ITERS:-0}" != 0 ]; then "$PYTHON" -I "$ROOT/bench/compare.py" "$OUT_DIR/long/ref" "$OUT_DIR/long/cand"; fi
