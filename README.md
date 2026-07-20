# Sprint Reconstruction — Keypoint Dataset

Produces two complementary keypoint datasets from monocular sprint videos, 30 fps,
for SMAS gait analysis.


---

## Quick Start

**If you only want to re-run the rollout optimization (which is my core coding part)** on existing `smpl_sequence.pkl`
files (if you already have those detected and converted from SAM3D), you can skip Step 1 entirely. You only need the
`sam_3d_body` conda env and the SMPL model.

```bash
conda activate sam_3d_body
export SMPL_MODEL=/path/to/SMPL_N_model_generate_from_npz.pkl
export SAM3D_DIR=/path/to/where/smpl_sequences/are   # locates smpl_sequences/
export DATASET_DIR=/path/to/sprint_videos             # needed for video fps detection

DEVICE=cuda bash scripts/run_rollout_all_athletes.sh          # all athletes
DEVICE=cuda bash scripts/run_rollout_all_athletes.sh Goree    # one athlete
```

Then skip to Step 3 (MHR fitting) and Step 4 (export JSON) below.

---

## Pipeline

```
Sprint video
  -> [Step 1]  SAM3D-Body detection + MHR->SMPL    scripts/run_sam3d_mhr_smpl.sh
  -> [Step 2]  Rollout optimization                 scripts/run_rollout_all_athletes.sh
  -> [Step 3]  MHR skeleton fitting                 scripts/run_mhr_kpts.sh
  -> [Step 4]  Keypoint export (2 datasets)          export/export_lowerbody.py
```

Step 3 runs `smpl_to_mhr_kpts.py` from the external MHR library via its pixi env.
Step 4 writes two JSON files per athlete: **v1** (SMPL-24, 2-D + orientation vectors)
and **v2** (MHR 21-joint lower-body, 3-D world + 2-D).

---

## Environment Setup

### Environment A — `sam_3d_body` conda (Steps 1, 2, 4)

```bash
# 1. Create env
conda env create -f environment.yml
conda activate sam_3d_body

# 2. Install PyTorch with the right CUDA version for your machine:
#    GPU:  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#    CPU:  pip install torch torchvision
#    See https://pytorch.org/get-started/locally/ for other versions.

# 3. Detectron2 (person detection in SAM3D)
pip install 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' \
    --no-build-isolation --no-deps

# 4. MoGe-2 (metric-scale focal length estimation)
pip install git+https://github.com/microsoft/MoGe.git
```

If you only need Steps 2 and 4 (re-running optimization or export on existing data) then you only need to run the following:
```bash
pip install -r requirements.txt   # minimal: torch, numpy, smplx, opencv, scipy
```

### Environment B — MHR pixi (Step 3 only)

```bash
# Install pixi package manager
curl -fsSL https://pixi.sh/install.sh | sh   # restart shell after

# Install MHR Python dependencies
cd <MHR_DIR>/tools/mhr_smpl_conversion
pixi add --pypi trimesh scikit-learn tqdm smplx
```

---

## Required Downloads

### 1. SAM3D-Body checkpoints

Request access at: https://huggingface.co/facebook/sam-3d-body-dinov3

Once approved:
```bash
cd <SAM3D_DIR>
huggingface-cli login
python -c "
from huggingface_hub import snapshot_download
snapshot_download('facebook/sam-3d-body-dinov3', local_dir='checkpoints/sam-3d-body-dinov3')
"
```

Expected files after download:
```
<SAM3D_DIR>/checkpoints/sam-3d-body-dinov3/model.ckpt
<SAM3D_DIR>/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt
```

### 2. SMPL body model

Register (free) at https://smpl.is.tue.mpg.de/ and download "SMPL for Python Users".
Extract to get `SMPL_NEUTRAL.npz`, then convert to pkl:

```bash
conda activate sam_3d_body
python tools/convert_smpl_npz_to_pkl.py \
    --input  /path/to/SMPL_NEUTRAL.npz \
    --output /path/to/smplhub/smpl/SMPL_N_model_generate_from_npz.pkl
```

Only the neutral model is required.

---

## Running the Pipeline

Set these env vars once (add to `~/.bashrc`):

```bash
export SAM3D_DIR=/path/to/sam-3d-body       # Step 1
export MHR_DIR=/path/to/MHR                 # Steps 1, 3
export SMPL_MODEL=/path/to/smplhub/smpl/SMPL_N_model_generate_from_npz.pkl  # Steps 1, 2, 3
export DATASET_DIR=/path/to/sprint_videos   # folder with <Athlete>.mp4 files (Steps 1, 4 tuner)
```

All scripts read these variables; no hardcoded paths in the code.

### Step 1 — SAM3D + MHR->SMPL

```bash
cd /path/to/Athlete_Mesh_Analysis_Codes
conda activate sam_3d_body

bash scripts/run_sam3d_mhr_smpl.sh Goree Bishop   # specific athletes
bash scripts/run_sam3d_mhr_smpl.sh                # all .mp4 in DATASET_DIR
DEVICE=cuda SKIP_EXISTING=1 bash scripts/run_sam3d_mhr_smpl.sh
```

The script handles everything: frame extraction (ffmpeg) → SAM3D detection + MoGe-2
focal length → target-athlete selection (`tools/extract_frames_for_conversion.py`) →
MHR→SMPL conversion via pixi (`tools/convert_mhr_to_smpl_and_rots_nrdf.py`) →
add SMPL 2D/3D keypoints (`tools/add_smpl_kpts_to_sequence.py`).

> Note: the pixi step still requires `cd $MHR_DIR/tools/mhr_smpl_conversion` to resolve
> the local `conversion.py` import. The shell script handles this automatically.

Output: `$SAM3D_DIR/outputs/smpl_sequences/<Athlete>/smpl_sequence.pkl`

> If the target athlete is not the largest bounding box, add an override to
> `$SAM3D_DIR/outputs/athlete_target_config.json` before running.

### Step 2 — Rollout optimization

```bash
conda activate sam_3d_body
DEVICE=cuda SKIP_EXISTING=1 bash scripts/run_rollout_all_athletes.sh Goree
```

Output: `outputs_with_moge2/rollout_results/<Athlete>/rollout.pkl`

The optimizer (`core/rollout_motion_optim.py`) represents the trajectory as translational
acceleration controls integrated via symplectic Euler, then minimises 2D reprojection
error + contact + lateral suppression + smoothness losses over 1000 iterations
(300 state-init phase + 700 full-control phase).

### Step 3 — MHR skeleton fitting (pixi)

```bash
conda deactivate   # pixi manages its own env — don't nest conda inside it

export MHR_DIR=... SMPL_MODEL=...
bash scripts/run_mhr_kpts.sh               # all athletes with rollout.pkl
bash scripts/run_mhr_kpts.sh Goree Walton  # specific athletes
SKIP_EXISTING=1 bash scripts/run_mhr_kpts.sh
```

This runs `smpl_to_mhr_kpts.py` from the MHR library (under `$MHR_DIR/tools/mhr_smpl_conversion/`)
via `pixi run`. It fits the 127-joint MHR skeleton to SMPL output and saves per-frame
joint positions plus bone rotation matrices for feet.

Output: `outputs_with_moge2/rollout_results/<Athlete>/mhr_kpts.pkl`

> Note: CPU only (~15–30 min per athlete). The MHR library's `pymomentum` backend
> was compiled against CPU-only PyTorch and segfaults if moved to CUDA.

### Step 4 — Keypoint export (two datasets)

```bash
cd /path/to/Athlete_Mesh_Analysis_Codes
conda activate sam_3d_body

python export/export_lowerbody.py                    # all athletes
python export/export_lowerbody.py --athlete Goree    # single athlete
```

One run writes **both** datasets:

| Dataset | Output path |
|---------|-------------|
| v1 — SMPL-24 | `sprint_smpl_dataset/data/<Athlete>.json` |
| v2 — MHR lower-body | `sprint_lowerbody_dataset/data/<Athlete>.json` |

v1 requires only `rollout.pkl`; v2 additionally requires `mhr_kpts.pkl` (Step 3).

---

## Output Format

### Dataset v1 — `sprint_smpl_dataset/data/<Athlete>.json`

24 standard SMPL joints, 30 fps.  Per-frame 2-D pixel coordinates only (no world
positions).  Also includes per-frame body orientation vectors.

```
Joint names (SMPL-24):
  pelvis
  left_hip, right_hip, spine1
  left_knee, right_knee, spine2
  left_ankle, right_ankle, spine3
  left_foot, right_foot, neck
  left_collar, right_collar, head
  left_shoulder, right_shoulder
  left_elbow, right_elbow
  left_wrist, right_wrist
  left_hand, right_hand
```

Per-frame fields (`frames[t]`):
- `kpts_2d` — `(24, 2)` pixel coordinates `[u_right, v_down]`
- `spine_up`, `chest_fwd`, `body_right` — unit 3-D vectors in Y-DOWN camera frame
- `facing_yaw_deg` — angle of chest_fwd (XZ plane) relative to sprint_dir
- `spine_vertical_deg` — angle of spine_up from world vertical

Top-level fields: `world_up` ([0,−1,0] = physical up in Y-DOWN frame), `sprint_dir`,
`joint_names`, `skeleton` (23 edges).

### Dataset v2 — `sprint_lowerbody_dataset/data/<Athlete>.json`

21 lower-body joints from the MHR skeleton (127-joint, step 3), 30 fps.
3-D world positions (Y-up, floor at Y=0, sprint ≈ +X) and 2-D pixel coordinates.

```
root
l/r_upleg, l/r_lowleg
l/r_foot,  l/r_heel_tip*,  l/r_heel_contact*
l/r_talocrural, l/r_subtalar, l/r_transversetarsal, l/r_ball, l/r_toe_tip*
```
`*` derived from MHR bone rotation matrices:
```
heel_tip(t)     = foot + R_foot @ POSTERIOR * 5.5 cm
heel_contact(t) = foot + R_foot @ POSTERIOR * 5.0 cm  +  R_foot @ INFERIOR * 11.0 cm
toe_tip(t)      = ball + R_ball @ DISTAL    * 3.5 cm
```

Per-joint fields (`joints[name]`):
- `world_m` — `(T, 3)`, Y-up metres, floor at Y=0
- `pixel_uv` — `(T, 2)`, pixel `[u_right, v_down]`, `null` if behind camera

Also contains `camera` (per-frame focal length, rotation, translation) and
`skeleton_edges` for visualisation.

### Re-tuning foot distances (v2)

```bash
cd export
python tune_heel.py --athlete Goree   # needs $DISPLAY and video in $DATASET_DIR
# z/x c/v b/n m/,  adjust distances  |  r: re-export JSON  |  w: save to script
```

---

## Debugging

```bash
python tools/inspect_pkl.py /path/to/rollout.pkl        # print keys and shapes
python tools/inspect_pkl.py /path/to/smpl_sequence.pkl
python tools/inspect_pkl.py /path/to/mhr_kpts.pkl
```

---

## File Structure

```
Athlete_Mesh_Analysis_Codes/
  core/
    rollout_motion_optim.py         <- Step 2: optimizer (self-contained)
    __init__.py
  export/
    export_lowerbody.py             <- Step 4: exports v1 (SMPL-24) + v2 (MHR lower-body)
    tune_heel.py                    <- interactive foot distance tuner
  scripts/
    run_sam3d_mhr_smpl.sh           <- Step 1: SAM3D + MHR->SMPL
    run_rollout_all_athletes.sh     <- Step 2: rollout batch runner
    run_mhr_kpts.sh                 <- Step 3: MHR skeleton fitting (pixi)
  tools/
    extract_frames_for_conversion.py   <- Step 1: isolate target athlete per frame
    convert_mhr_to_smpl_and_rots_nrdf.py <- Step 1: MHR vertices → SMPL params (pixi)
    add_smpl_kpts_to_sequence.py       <- Step 1: add SMPL 2D/3D keypoints to pkl
    convert_smpl_npz_to_pkl.py         <- one-time: convert SMPL .npz to .pkl
    inspect_pkl.py                     <- debug: print pkl keys and shapes
  environment.yml                   <- conda env for Steps 1, 2, 4
  requirements.txt                  <- minimal pip deps for Steps 2 + 4 only
  README.md
```
