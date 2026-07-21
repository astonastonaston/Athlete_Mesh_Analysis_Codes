#!/usr/bin/env bash
# run_mhr_kpts.sh
#
# Step 3: MHR skeleton fitting.
# Reads rollout.pkl → fits MHR 127-joint skeleton → writes mhr_kpts.pkl.
# mhr_kpts.pkl contains per-frame joint positions AND bone rotation matrices
# for l_foot, r_foot, l_ball, r_ball — used by Step 4 (export_lowerbody.py)
# to derive the heel_tip, heel_contact, and toe_tip landmarks.
#
# Runs smpl_to_mhr_kpts.py via pixi (MHR's env). CPU only (~15-30 min/athlete).
#
# Required env vars:
#   MHR_DIR     path to MHR repo
#   SMPL_MODEL  path to SMPL_N_model_generate_from_npz.pkl
#
# Usage:
#   export MHR_DIR=... SMPL_MODEL=...
#   cd /path/to/Athlete_Mesh_Analysis_Codes
#
#   bash scripts/run_mhr_kpts.sh               # all athletes with rollout.pkl
#   bash scripts/run_mhr_kpts.sh Goree Walton  # specific athletes
#   SKIP_EXISTING=1 bash scripts/run_mhr_kpts.sh
# -----------------------------------------------------------------------------

set -euo pipefail

for var in MHR_DIR SMPL_MODEL; do
    if [ -z "${!var:-}" ]; then
        echo "[ERROR] \$$var is not set. Export it before running."
        exit 1
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODES_DIR="$(dirname "${SCRIPT_DIR}")"
NRMF_DIR="$(dirname "${CODES_DIR}")"

ROLLOUT_ROOT="${ROLLOUT_ROOT:-${NRMF_DIR}/outputs_with_moge2/rollout_results}"
MHR_SCRIPT="${MHR_DIR}/tools/mhr_smpl_conversion/smpl_to_mhr_kpts.py"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

if [ ! -f "${MHR_SCRIPT}" ]; then
    echo "[ERROR] smpl_to_mhr_kpts.py not found: ${MHR_SCRIPT}"
    echo "  Copy it first:  cp ${CODES_DIR}/mhr_tools/smpl_to_mhr_kpts.py \${MHR_DIR}/tools/mhr_smpl_conversion/"
    exit 1
fi

# --- athlete list ------------------------------------------------------------
if [ $# -ge 1 ]; then
    ATHLETES=("$@")
else
    mapfile -t ATHLETES < <(
        for d in "${ROLLOUT_ROOT}"/*/rollout.pkl; do
            [ -f "$d" ] || continue
            basename "$(dirname "$d")"
        done | sort
    )
fi

if [ ${#ATHLETES[@]} -eq 0 ]; then
    echo "[ERROR] No rollout.pkl found under ${ROLLOUT_ROOT}"
    echo "  Run run_rollout_all_athletes.sh first."
    exit 1
fi

echo "===================================================="
echo "MHR skeleton fitting  |  athletes: ${ATHLETES[*]}"
echo "  rollout_root : ${ROLLOUT_ROOT}"
echo "  skip_existing: ${SKIP_EXISTING}"
echo "===================================================="

FAILED=()

for ATH in "${ATHLETES[@]}"; do
    IN_PKL="${ROLLOUT_ROOT}/${ATH}/rollout.pkl"
    OUT_PKL="${ROLLOUT_ROOT}/${ATH}/mhr_kpts.pkl"

    echo ""
    echo "---- [${ATH}] -----------------------------------------------"

    if [ ! -f "${IN_PKL}" ]; then
        echo "  [WARN] rollout.pkl not found: ${IN_PKL} -- skipping"
        FAILED+=("${ATH}:no_input"); continue
    fi

    if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${OUT_PKL}" ]; then
        echo "  [skip] mhr_kpts.pkl already exists"
        continue
    fi

    ( cd "${MHR_DIR}/tools/mhr_smpl_conversion" && \
      pixi run python smpl_to_mhr_kpts.py \
          --athlete         "${ATH}" \
          --out             "${OUT_PKL}" \
          --rollout_dir     "${ROLLOUT_ROOT}" \
          --smpl_model_path "${SMPL_MODEL}" \
    ) || { echo "  [ERROR] MHR fitting failed"; FAILED+=("${ATH}:mhr"); continue; }

    echo "  [OK] ${OUT_PKL}"
done

echo ""
echo "===================================================="
for ATH in "${ATHLETES[@]}"; do
    OUT_PKL="${ROLLOUT_ROOT}/${ATH}/mhr_kpts.pkl"
    if [ -f "${OUT_PKL}" ]; then
        echo "  [OK]     ${ATH} -> $(du -h "${OUT_PKL}" | cut -f1)"
    else
        echo "  [FAILED] ${ATH}"
    fi
done
[ ${#FAILED[@]} -gt 0 ] && echo "  failed: ${FAILED[*]}"
echo "===================================================="
echo ""
echo "Next: cd export && python export_lowerbody.py"
echo "  (derives heel_tip / heel_contact / toe_tip from mhr_kpts.pkl bone rotations)"
