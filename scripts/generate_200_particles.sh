#!/usr/bin/env bash

# Generate a reproducible, balanced 200-particle candidate database.
#
# The ranges are centred on the example case:
#   e=0.75, f=0.65, taper=0.25, boxiness=3.0,
#   asymmetry-I=0.12, asymmetry-S=-0.06, randomness=0.10.
#
# There are 50 cases for each family: ellipsoid, tapered_ellipsoid,
# superellipsoid, and rounded_trapezoid. Taper and boxiness are automatically
# ignored for families that do not use them. The asymmetry magnitudes are
# sampled independently in [0, 0.15], with an independently random sign.
#
# sub=1, div=50, and 2,000 Monte Carlo samples match the supplied command and
# are intended for coarse screening. Increase them for a final database.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_hime_gravel_database.sh"

if [[ ! -f "${RUNNER}" ]]; then
    echo "[ERROR] Batch runner not found: ${RUNNER}" >&2
    exit 2
fi

# Physical source-shape controls varied from case to case.
export HIME_L_MIN="${HIME_L_MIN:-1.0}"
export HIME_L_MAX="${HIME_L_MAX:-1.0}"
export HIME_E_MIN="${HIME_E_MIN:-0.70}"
export HIME_E_MAX="${HIME_E_MAX:-0.80}"
export HIME_F_MIN="${HIME_F_MIN:-0.60}"
export HIME_F_MAX="${HIME_F_MAX:-0.70}"
export HIME_TAPER_MIN="${HIME_TAPER_MIN:-0.15}"
export HIME_TAPER_MAX="${HIME_TAPER_MAX:-0.35}"
export HIME_BOXINESS_MIN="${HIME_BOXINESS_MIN:-2.5}"
export HIME_BOXINESS_MAX="${HIME_BOXINESS_MAX:-3.5}"
export HIME_ASYM_MIN="${HIME_ASYM_MIN:-0.0}"
export HIME_ASYM_MAX="${HIME_ASYM_MAX:-0.15}"
export HIME_RANDOMNESS_MIN="${HIME_RANDOMNESS_MIN:-0.10}"
export HIME_RANDOMNESS_MAX="${HIME_RANDOMNESS_MAX:-0.15}"

# Bias was not requested as a varying control, so it remains fixed at 1.5.
export HIME_BIAS_MIN="${HIME_BIAS_MIN:-1.5}"
export HIME_BIAS_MAX="${HIME_BIAS_MAX:-1.5}"

# Fixed source-mesh, clump, and Monte Carlo settings from the supplied command.
export HIME_SUBDIVISIONS="${HIME_SUBDIVISIONS:-1}"
export HIME_N="${HIME_N:-60}"
export HIME_RMIN="${HIME_RMIN:-0.0}"
export HIME_DIV="${HIME_DIV:-50}"
export HIME_OVERLAP="${HIME_OVERLAP:-0.95}"
export HIME_RMAX_RATIO="${HIME_RMAX_RATIO:-1.0}"
export HIME_GS="${HIME_GS:-2.65}"
export HIME_SAMPLES_VOLUME="${HIME_SAMPLES_VOLUME:-2000}"
export HIME_SAMPLES_INERTIA="${HIME_SAMPLES_INERTIA:-2000}"
export HIME_MAKE_MOLECULE="${HIME_MAKE_MOLECULE:-1}"

MASTER_SEED="${PARTICLE_MASTER_SEED:-1234}"

exec bash "${RUNNER}" \
    --cases-per-family 50 \
    --seed "${MASTER_SEED}" \
    --dataset-dir dataset/particles_200/shapes \
    --manifest dataset/particles_200/manifest.jsonl \
    "$@"
