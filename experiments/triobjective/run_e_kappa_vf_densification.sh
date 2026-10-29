#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_CSV="${SOURCE_CSV:-}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/data/results/triobjective-e-kappa-vf}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -z "${SOURCE_CSV}" ]]; then
  echo "SOURCE_CSV is required. Use a local weighted CSV with multiple volume fractions." >&2
  exit 2
fi
if [[ ! -f "${SOURCE_CSV}" ]]; then
  echo "Source CSV not found: ${SOURCE_CSV}" >&2
  echo "Set SOURCE_CSV to a varied-vf CSV containing weight columns." >&2
  exit 2
fi
if ! head -n 1 "${SOURCE_CSV}" | grep -Eq '(^|,)(w_stiffness|w_thermal|objective)(,|$)'; then
  echo "Source CSV has no weight columns: ${SOURCE_CSV}" >&2
  echo "Paper-style e_kappa_vf interpolation requires source objective weights." >&2
  exit 2
fi
CMD=("${PYTHON_BIN}" -m maple.pareto.run_iterative_property_network_densification
  --initial_source_csv "${SOURCE_CSV}"
  --output_root "${OUT_ROOT}"
  --front_objectives e_kappa_vf
  --sampling_mode adaptive_front
  --vf_objective minimize
  --tri_weight_mode source
  --rounds "${ROUNDS:-5}"
  --samples_per_round "${SAMPLES_PER_ROUND:-200}"
  --lambda_print "${LAMBDA_PRINT:-0.9}"
  --overhang_deg "${OVERHANG_DEG:-45}"
  --lambda_cavity "${LAMBDA_CAVITY:-0.9}"
  --min_channel_rad "${MIN_CHANNEL_RAD:-3}"
  --cavity_tiles "${CAVITY_TILES:-2}"
  --gpus "${GPUS:-auto}")

if [[ -n "${STAGE2_CHECKPOINT:-}" ]]; then
  CMD+=(--initial_checkpoint "${STAGE2_CHECKPOINT}")
fi

exec "${CMD[@]}" "$@"
