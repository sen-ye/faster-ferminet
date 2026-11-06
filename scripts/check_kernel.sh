#!/usr/bin/env bash
# Kernel contract probe on one GPU: Pallas trsm vs the real cuBLAS path, bit for bit,
# n in 9..64, all four lu_solve variants, three input scales; plus kernel timings.
# Run this first on a new GPU / CUDA / JAX stack (see "Scope" in README.md).
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} JAX_PLATFORMS=cuda \
  "$PYTHON" -I "$ROOT/bench/trsm_check.py" "$ROOT" "${SIZES:-9,16,17,26,32,33,36,48,54,64}"
