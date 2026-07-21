#!/usr/bin/env python3
"""
smpl_to_mhr_kpts.py
────────────────────
Load a Module-1 rollout.pkl (SMPL world-frame params) → run SMPL forward →
convert SMPL vertices to MHR via Conversion.convert_smpl2mhr() → extract
detailed foot-chain joint positions (ankle, talocrural, subtalar,
transversetarsal, ball) from the MHR skeleton state.

The ball-of-foot joints (l_ball / r_ball) are the metatarsophalangeal
joints — i.e., the "toe knuckle" where the foot flexes during push-off.
MHR decomposes the foot into 5 joints per side vs SMPL's single foot joint.

Usage (must run from $MHR_DIR/tools/mhr_smpl_conversion; uses pixi env Python):

  cd $MHR_DIR/tools/mhr_smpl_conversion

  # Single athlete (CPU, ~15-30 min):
  $MHR_DIR/.pixi/envs/default/bin/python smpl_to_mhr_kpts.py \\
      --athlete Original \\
      --rollout_dir $NRMF_ROLLOUT_DIR \\
      --smpl_model_path $SMPL_MODEL \\
      --out $NRMF_ROLLOUT_DIR/Original/mhr_kpts.pkl

  # All athletes:
  for ATH in Bishop Colin_Brazzell Goree Jackson Original Poteat Walton; do
    $MHR_DIR/.pixi/envs/default/bin/python smpl_to_mhr_kpts.py \\
        --athlete $ATH \\
        --rollout_dir $NRMF_ROLLOUT_DIR \\
        --smpl_model_path $SMPL_MODEL \\
        --out $NRMF_ROLLOUT_DIR/$ATH/mhr_kpts.pkl
  done

LOD NOTE: Must use lod=1 (default). The SMPL→MHR vertex mapping (smpl2mhr_mapping.npz)
and face mask (mhr_face_mask.ply) both have 18439 vertices = lod1 mesh size. Using
lod3 (4899 verts) causes a RuntimeError at pytorch_fitting.py:978. Foot joint positions
come from skeleton FK so the LOD doesn't affect their accuracy.

CUDA: pymomentum in the pixi env is compiled against CPU-only PyTorch (ABI constraint).
Moving MHR to CUDA segfaults. SMPL forward can run on CUDA but MHR fitting stays CPU.

Output pkl keys:
  athlete                       str
  n_frames                      int
  joint_names                   list[str]  — all MHR joint names (116 total)
  all_joint_positions_world_m   np.ndarray (T, J, 3)  meters, SAM3D world frame
  foot_joint_positions_world_m  dict[str → np.ndarray (T, 3)]
      Keys: l_foot, l_talocrural, l_subtalar, l_transversetarsal, l_ball,
            r_foot, r_talocrural, r_subtalar, r_transversetarsal, r_ball
  fit_errors_cm                 np.ndarray (T,)  — per-frame SMPL→MHR vertex error (cm)
"""
import argparse
import os
import pickle
import time

import numpy as np
import torch
import smplx
from pathlib import Path

from mhr.mhr import MHR
from conversion import Conversion
import pymomentum.skel_state as skel_state_ops


# ── Paths (defaults; overridden by CLI args --rollout_dir / --smpl_model_path) ─
ROLLOUT_DIR = os.environ.get("NRMF_ROLLOUT_DIR",
              "/path/to/outputs_with_moge2/rollout_results")
SMPL_MODEL  = os.environ.get("SMPL_MODEL",
              "/path/to/smplhub/smpl/SMPL_N_model_generate_from_npz.pkl")
MHR_ASSETS  = str(Path(__file__).resolve().parents[2] / "assets")

ATHLETES = ["Bishop", "Colin_Brazzell", "Goree", "Jackson", "Original", "Poteat", "Walton"]

FOOT_JOINTS = [
    "l_foot", "l_talocrural", "l_subtalar", "l_transversetarsal", "l_ball",
    "r_foot", "r_talocrural", "r_subtalar", "r_transversetarsal", "r_ball",
]


def _blendshapes_mb(lod: int) -> float:
    p = Path(MHR_ASSETS) / f"corrective_blendshapes_lod{lod}.npz"
    if p.exists():
        return p.stat().st_size / 1e6
    return -1.0


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--athlete",          required=True, choices=ATHLETES)
    ap.add_argument("--out",              required=True, help="Output pkl path")
    ap.add_argument("--rollout_dir",      default=ROLLOUT_DIR,
                    help="Directory containing <Athlete>/rollout.pkl files.")
    ap.add_argument("--smpl_model_path",  default=SMPL_MODEL,
                    help="Path to SMPL neutral model .pkl file.")
    ap.add_argument("--device",       default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--method",       default="pytorch", choices=["pytorch", "pymomentum"],
                    help="Fitting backend. pytorch supports GPU; pymomentum is CPU-only.")
    ap.add_argument("--lod",          type=int, default=1,
                    help="MHR level-of-detail. Must be 1 for SMPL→MHR conversion "
                         "(conversion code requires lod1's 18439-vertex mesh). "
                         "lod1 blendshapes = 634MB but loads in ~30s.")
    ap.add_argument("--no_tracking",  action="store_true",
                    help="Fit each frame independently (slower). Default: temporal init.")
    args = ap.parse_args()
    is_tracking = not args.no_tracking

    # ── Device setup ──────────────────────────────────────────────────────────
    # pymomentum (MHR's C++ backend) was compiled against CPU-only torch in the
    # pixi env. Moving MHR to CUDA segfaults due to ABI mismatch.
    # Fix: SMPL forward runs on GPU (pure PyTorch, fast); MHR stays on CPU.
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False.\n"
                "Fix: $MHR_DIR/.pixi/envs/default/bin/python -m pip install torch "
                "--index-url https://download.pytorch.org/whl/cu128 "
                "--force-reinstall --no-deps"
            )
        smpl_device = torch.device("cuda:0")
    else:
        smpl_device = torch.device("cpu")
    mhr_device = torch.device("cpu")   # pymomentum must stay on CPU
    print(f"SMPL device: {smpl_device}  |  MHR device: {mhr_device}")

    # ── 1. Load rollout pkl ───────────────────────────────────────────────────
    pkl_path = os.path.join(args.rollout_dir, args.athlete, "rollout.pkl")
    print(f"Loading: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    sp = data["smpl_parameters"]

    T      = len(sp["transl_world_refined"])
    transl = torch.tensor(sp["transl_world_refined"],        dtype=torch.float32)
    go     = torch.tensor(sp["global_orient_world_refined"], dtype=torch.float32)
    bp     = torch.tensor(sp["body_pose"],                   dtype=torch.float32)
    betas  = torch.tensor(sp["betas"],                       dtype=torch.float32)
    print(f"  {args.athlete}: T={T} frames")

    # ── 2. SMPL forward → world-frame vertices (meters, on GPU if available) ──
    print("Running SMPL forward pass...")
    smpl_model = smplx.SMPL(model_path=args.smpl_model_path, num_betas=10, batch_size=T).to(smpl_device)
    transl = transl.to(smpl_device); go = go.to(smpl_device)
    bp     = bp.to(smpl_device);     betas = betas.to(smpl_device)
    with torch.no_grad():
        smpl_out = smpl_model(global_orient=go, body_pose=bp, betas=betas, transl=transl)
    smpl_verts_gpu = smpl_out.vertices   # (T, 6890, 3) on smpl_device
    smpl_verts = smpl_verts_gpu.cpu()   # move to CPU for MHR (pymomentum requires CPU)
    print(f"  SMPL verts: {smpl_verts.shape}  Y range: "
          f"[{smpl_verts[:,:,1].min():.3f}, {smpl_verts[:,:,1].max():.3f}] m")

    # ── 3. Init MHR + converter (always CPU — pymomentum ABI constraint) ──────
    mb = _blendshapes_mb(args.lod)
    print(f"Initializing MHR model (lod={args.lod}, blendshapes={mb:.0f} MB, device=cpu)...")
    t0 = time.time()
    mhr_model = MHR.from_files(lod=args.lod, folder=Path(MHR_ASSETS), device=mhr_device)
    print(f"  MHR loaded in {time.time()-t0:.1f}s")
    smpl_model_cpu = smplx.SMPL(model_path=args.smpl_model_path, num_betas=10, batch_size=T)
    converter = Conversion(mhr_model=mhr_model, smpl_model=smpl_model_cpu, method=args.method)

    # ── 4. SMPL vertices → MHR parameters ────────────────────────────────────
    # single_identity=True: shared body shape across all frames (more stable)
    # is_tracking=True: use prev frame as init for next → faster convergence
    print(f"Converting SMPL → MHR  (T={T}, tracking={is_tracking})...")
    t0 = time.time()
    result = converter.convert_smpl2mhr(
        smpl_vertices=smpl_verts,
        single_identity=True,
        is_tracking=is_tracking,
        return_mhr_parameters=True,
        return_mhr_vertices=False,
        return_mhr_meshes=False,
        return_fitting_errors=True,
    )
    print(f"  Conversion done in {time.time()-t0:.1f}s")
    mhr_params = result.result_parameters
    fit_errors = result.result_errors  # (T,) in cm
    if fit_errors is not None:
        print(f"  Fit errors: mean={np.mean(fit_errors):.3f} cm  "
              f"max={np.max(fit_errors):.3f} cm")

    # ── 5. MHR forward → skeleton state (CPU) ────────────────────────────────
    print("Running MHR forward pass to get skeleton state...")
    with torch.no_grad():
        _, skel_state = mhr_model(
            identity_coeffs=mhr_params["identity_coeffs"].to(mhr_device),
            model_parameters=mhr_params["lbs_model_params"].to(mhr_device),
            face_expr_coeffs=mhr_params["face_expr_coeffs"].to(mhr_device),
            apply_correctives=True,
        )

    # ── 6. Extract joint world positions ─────────────────────────────────────
    # skel_state_ops.split → (translation_cm, rotation_quat, scale)
    trans_cm = skel_state_ops.split(skel_state)[0]
    trans_m = trans_cm.cpu().numpy() * 0.01   # cm → meters

    jnames = mhr_model.character_torch.skeleton.joint_names

    foot_kpts = {}
    for jname in FOOT_JOINTS:
        if jname in jnames:
            idx = list(jnames).index(jname)
            foot_kpts[jname] = trans_m[:, idx, :]   # (T, 3) meters

    print(f"  Extracted foot joints: {list(foot_kpts.keys())}")
    if "l_ball" in foot_kpts:
        print(f"  l_ball Y range: [{foot_kpts['l_ball'][:,1].min():.3f}, "
              f"{foot_kpts['l_ball'][:,1].max():.3f}] m")

    # ── 6b. Extract bone rotation matrices for heel/toe tip computation ───────
    # skel_state_ops.to_matrix → (T, J, 4, 4) world transforms; [:,:,:3,:3] = rotations.
    # These are in the same world frame as trans_m (SAM3D Y-DOWN).
    # Saved for l_foot (calcaneus), r_foot, l_ball (MTP), r_ball.
    print("Extracting bone rotation matrices...")
    with torch.no_grad():
        bone_mats = skel_state_ops.to_matrix(skel_state)   # (T, J, 4, 4)
    bone_mats_np = bone_mats.cpu().numpy()

    BONE_ROT_JOINTS = ["l_foot", "r_foot", "l_ball", "r_ball"]
    bone_rots = {}
    for jname in BONE_ROT_JOINTS:
        if jname in jnames:
            idx = list(jnames).index(jname)
            bone_rots[f"{jname}_world_rot"] = bone_mats_np[:, idx, :3, :3]  # (T,3,3)
    print(f"  Bone rotations saved: {list(bone_rots.keys())}  shape={bone_mats_np.shape[:2]}")

    # ── 7. Save ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out_data = {
        "athlete":    args.athlete,
        "n_frames":   T,
        "joint_names": list(jnames),
        "all_joint_positions_world_m":  trans_m,
        "foot_joint_positions_world_m": foot_kpts,
        "fit_errors_cm": fit_errors,
    }
    out_data.update(bone_rots)   # adds l_foot_world_rot, r_foot_world_rot, l_ball_world_rot, r_ball_world_rot
    with open(args.out, "wb") as f:
        pickle.dump(out_data, f)
    print(f"\nSaved → {args.out}")
    for k, v in out_data.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape}")
        elif isinstance(v, dict):
            print(f"  {k}: {list(v.keys())}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
