#!/usr/bin/env bash
# run_rollout_all_athletes.sh
#
# Physics-constrained trajectory rollout optimization.
# Reads  smpl_sequence.pkl  from the SAM3D+MHR->SMPL pipeline.
# Writes rollout.pkl to OUT_ROOT/<Athlete>/.
#
# Required env vars (set once, or add to ~/.bashrc):
#   SAM3D_DIR   path to sam-3d-body repo  (locates smpl_sequences/)
#   SMPL_MODEL  path to SMPL_N_model_generate_from_npz.pkl
#   DATASET_DIR folder with <Athlete>.mp4 files
#               NOTE: use 1920x1080 single-athlete crops, NOT panoramic 7680x1080 video.
#               The panoramic crop gives cx=3840 which causes a -14m world-X shift.
#
# Usage:
#   export SAM3D_DIR=... SMPL_MODEL=... DATASET_DIR=...
#   cd /path/to/Athlete_Mesh_Analysis_Codes
#
#   bash scripts/run_rollout_all_athletes.sh               # all athletes in SEQ_ROOT
#   bash scripts/run_rollout_all_athletes.sh Goree Walton  # specific athletes
#   DEVICE=cuda SKIP_EXISTING=1 bash scripts/run_rollout_all_athletes.sh
# -----------------------------------------------------------------------------

set -euo pipefail

# --- validate required env vars ----------------------------------------------
for var in SAM3D_DIR SMPL_MODEL DATASET_DIR; do
    if [ -z "${!var:-}" ]; then
        echo "[ERROR] \$$var is not set. Export it before running."
        exit 1
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODES_DIR="$(dirname "${SCRIPT_DIR}")"
NRMF_DIR="$(dirname "${CODES_DIR}")"

SEQ_ROOT="${SEQ_ROOT:-${SAM3D_DIR}/outputs/smpl_sequences}"
VIDEO_ROOT="${VIDEO_ROOT:-${DATASET_DIR}}"
OUT_ROOT="${OUT_ROOT:-${NRMF_DIR}/outputs_with_moge2/rollout_results}"
ROLLOUT_SCRIPT="${CODES_DIR}/core/rollout_motion_optim.py"

DEVICE="${DEVICE:-cpu}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
CONDA_ENV="${CONDA_ENV:-sam_3d_body}"

# --- athlete list ------------------------------------------------------------
if [ $# -ge 1 ]; then
    ATHLETES=("$@")
else
    mapfile -t ATHLETES < <(
        for d in "${SEQ_ROOT}"/*/smpl_sequence.pkl; do
            [ -f "$d" ] || continue
            basename "$(dirname "$d")"
        done | sort
    )
fi

if [ ${#ATHLETES[@]} -eq 0 ]; then
    echo "[ERROR] No smpl_sequence.pkl found under ${SEQ_ROOT}"
    echo "  Run run_sam3d_mhr_smpl.sh first."
    exit 1
fi

echo "===================================================="
echo "Rollout optimization  |  athletes: ${ATHLETES[*]}"
echo "  out_root : ${OUT_ROOT}"
echo "  device   : ${DEVICE}  |  skip_existing: ${SKIP_EXISTING}"
echo "===================================================="

FAILED=()

for ATH in "${ATHLETES[@]}"; do
    IN_PKL="${SEQ_ROOT}/${ATH}/smpl_sequence.pkl"
    OUT_PKL="${OUT_ROOT}/${ATH}/rollout.pkl"
    VIDEO="${VIDEO_ROOT}/${ATH}.mp4"

    echo ""
    echo "---- [${ATH}] -----------------------------------------------"

    if [ ! -f "${IN_PKL}" ]; then
        echo "  [WARN] smpl_sequence.pkl not found: ${IN_PKL} -- skipping"
        FAILED+=("${ATH}:no_input"); continue
    fi

    if [ "${SKIP_EXISTING}" = "1" ] && [ -f "${OUT_PKL}" ]; then
        echo "  [skip] rollout.pkl already exists"
        continue
    fi

    mkdir -p "$(dirname "${OUT_PKL}")"
    VIDEO_ARG=""
    [ -f "${VIDEO}" ] && VIDEO_ARG="--video ${VIDEO}"

    conda run --no-capture-output -n "${CONDA_ENV}" python "${ROLLOUT_SCRIPT}" \
        --in_pkl          "${IN_PKL}" \
        --smpl_model_path "${SMPL_MODEL}" \
        --out_pkl         "${OUT_PKL}" \
        --device          "${DEVICE}" \
        ${VIDEO_ARG} \
        --iters             1000 \
        --stage1_iters      300 \
        --debug_every       200 \
        --sprint_dir_mode   sam3d_orient \
        --orient_mode       yaw_sam_pitchroll \
        --use_pulse_model   0 \
        --opt_cam           0 \
        --w_reproj          1.0 \
        --w_reproj_stage1   5.0 \
        --w_lat             1.0 \
        --w_smooth_fwd      0.05 \
        --w_yaw_ctrl        0.05 \
        --w_anchor          0.05 \
        --w_anchor_yaw      0.01 \
        --use_contact_loss  1 \
        --w_contact         1.0 \
        --normalize_lat_loss 1 \
        --grad_clip         1.0 \
        --smooth_a_ctrl_sigma 3.0 \
        || { echo "  [ERROR] rollout failed"; FAILED+=("${ATH}:rollout"); continue; }

    echo "  [OK] ${OUT_PKL}"
done

# --- summary -----------------------------------------------------------------
echo ""
echo "===================================================="
for ATH in "${ATHLETES[@]}"; do
    OUT_PKL="${OUT_ROOT}/${ATH}/rollout.pkl"
    if [ -f "${OUT_PKL}" ]; then
        echo "  [OK]     ${ATH} -> $(du -h "${OUT_PKL}" | cut -f1)"
    else
        echo "  [FAILED] ${ATH}"
    fi
done
[ ${#FAILED[@]} -gt 0 ] && echo "  failed: ${FAILED[*]}"
echo "===================================================="
echo ""
echo "Next: run MHR skeleton fitting, then:"
echo "  cd export && python export_lowerbody.py"
