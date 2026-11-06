#!/usr/bin/env bash
# Formal benchmark (the protocol of the published numbers), one atom per GPU:
#   reference A -> Faster-FermiNet -> reference B timing (default XLA flags), and
#   deterministic-mode bit-wise records of both trees, then the gate verdict.
# Env: GPUS (4 GPU ids, default 0,1,2,3), PYTHON, REFERENCE_TREE, OUT_DIR,
#      LONG_ITERS (e.g. 110 for the long bit-wise check; default 0 = off), PROBE=1.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
GPUS=${GPUS:-0,1,2,3}
REFERENCE_TREE=${REFERENCE_TREE:-$ROOT/third_party/ferminet-upstream}
OUT_DIR=${OUT_DIR:-$ROOT/runs/benchmark-$(date +%Y%m%d-%H%M%S)}
LONG_ITERS=${LONG_ITERS:-0}
[ -d "$REFERENCE_TREE/ferminet" ] || { echo "missing reference tree; run scripts/get_upstream.sh" >&2; exit 1; }
source "$ROOT/scripts/configs/faster.env"
EXTRA=(); [ "${PROBE:-0}" = 1 ] && EXTRA+=(--probe)
"$PYTHON" "$ROOT/bench/make_jobs.py" --plan gate --reference "$REFERENCE_TREE" --candidate "$ROOT" \
  --out "$OUT_DIR" --gpus "$GPUS" --long-iters "$LONG_ITERS" "${EXTRA[@]}"
PYTHON="$PYTHON" "$PYTHON" "$ROOT/bench/launch.py" "$OUT_DIR/jobs.json" "$GPUS"
"$PYTHON" "$ROOT/bench/gate.py" "$OUT_DIR" | tee "$OUT_DIR/final_table.md"
