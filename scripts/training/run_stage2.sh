#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="${1:-${ROOT}/data/results/stage2_data/manifest.csv}"
OUT_DIR="${2:-${ROOT}/data/results/stage2_runs/density_generator}"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

python -m maple.training.train_stage2_density_generator \
  --manifest "${MANIFEST}" \
  --output_dir "${OUT_DIR}" \
  --target_key rho_target_32 \
  --epochs 200 \
  --batch_size 4 \
  --num_workers 4 \
  --lr 2.0e-4 \
  --weight_decay 1.0e-4 \
  --alpha_vol 1.0 \
  --beta_smooth 1.0e-3 \
  --amp
