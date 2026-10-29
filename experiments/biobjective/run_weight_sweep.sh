#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/data/results/biobjective-weight-sweep}"
VF="${VF:-0.25}"
SEEDS_PER_W="${SEEDS_PER_W:-10}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON_BIN}" -m maple.data.build_weight_sweep_dataset \
  --output_root "${OUT_ROOT}" \
  --volfrac "${VF}" \
  --seeds_per_w "${SEEDS_PER_W}" \
  --lambda_print "${LAMBDA_PRINT:-0.9}" \
  --overhang_deg "${OVERHANG_DEG:-45}" \
  --lambda_cavity "${LAMBDA_CAVITY:-0.9}" \
  --min_channel_rad "${MIN_CHANNEL_RAD:-3}" \
  --cavity_tiles "${CAVITY_TILES:-2}" \
  "$@"
