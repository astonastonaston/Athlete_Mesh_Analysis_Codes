#!/usr/bin/env python3
"""
add_smpl_kpts_to_sequence.py

Post-processing step after MHR→SMPL conversion.
Runs an SMPL forward pass on the converted sequence, projects 24 joints
to 2D, and writes smpl_kpts2d_24 / smpl_kpts2d_21 into the output pkl.

Optionally merges MHR 2D/3D keypoints from a sequence_mhr.pkl so the
final pkl has everything needed for jitter analysis and the rollout pipeline.

Run with sam_3d_body conda env (has smplx, torch).

Usage
-----
conda run -n sam_3d_body python tools/add_smpl_kpts_to_sequence.py \
    --smpl_pkl  outputs/smpl_sequences/Amari/smpl_raw.pkl \
    --smpl_model_path /path/to/SMPL_N_model_generate_from_npz.pkl \
    --out_pkl   outputs/smpl_sequences/Amari/smpl_sequence.pkl

# Also merge MHR 2D/3D kpts:
    --mhr_seq_pkl outputs/sequences/Amari/sequence_mhr.pkl

# cx/cy if known (default 0):
    --cx 960.0 --cy 540.0
"""

import argparse, os, pickle, sys
import numpy as np

try:
    import torch
    import smplx
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[ERROR] torch / smplx required."); sys.exit(1)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pkl(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)


def run_smpl_forward(smpl_params, smpl_model, device, chunk=256):
    """
    smpl_params : dict with numpy arrays global_orient (T,3), body_pose (T,69),
                  betas (T,10), transl_final (T,3)
    Returns joints (T, 24, 3) numpy, world frame.
    """
    def _get(d, *keys):
        for k in keys:
            if k in d and d[k] is not None:
                v = d[k]
                return v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
        raise KeyError(f"None of {keys} found in smpl_params")

    go    = _get(smpl_params, "global_orient", "root_orient").astype(np.float32)
    bp    = _get(smpl_params, "body_pose", "pose_body").astype(np.float32)
    betas = _get(smpl_params, "betas", "shape").astype(np.float32)
    tr    = _get(smpl_params, "transl_final", "transl").astype(np.float32)

    T = go.shape[0]
    if betas.shape[0] == 1:
        betas = np.tile(betas, (T, 1))
    betas = betas[:T, :10]
    bp    = bp.reshape(T, -1)[:, :69]

    outs = []
    for i in range(0, T, chunk):
        sl = slice(i, i + chunk)
        with torch.no_grad():
            out = smpl_model(
                global_orient = torch.tensor(go[sl],    device=device).float(),
                body_pose     = torch.tensor(bp[sl],    device=device).float(),
                betas         = torch.tensor(betas[sl], device=device).float(),
                transl        = torch.tensor(tr[sl],    device=device).float(),
                return_verts  = False,
                pose2rot      = True,
            )
        outs.append(out.joints[:, :24, :].cpu().numpy())
    return np.concatenate(outs, axis=0).astype(np.float32)   # (T, 24, 3)


def project_joints_2d(joints_3d, focal, cx=0.0, cy=0.0):
    """
    joints_3d : (T, J, 3) world/camera frame
    focal     : (T,) or scalar
    Returns   : (T, J, 2) pixel coords
    Convention: cam is identity (world = camera frame), perspective:
                u = f * X / Z + cx
    """
    T = joints_3d.shape[0]
    if np.isscalar(focal) or np.asarray(focal).ndim == 0:
        focal = np.full(T, float(focal), dtype=np.float32)
    focal = np.asarray(focal, dtype=np.float32).reshape(T)

    Z = np.clip(joints_3d[..., 2:3], 1e-3, None)
    u = focal[:, None, None] * joints_3d[..., 0:1] / Z + cx
    v = focal[:, None, None] * joints_3d[..., 1:2] / Z + cy
    return np.concatenate([u, v], axis=-1).astype(np.float32)   # (T, J, 2)


def main():
    p = argparse.ArgumentParser(
        description="Add SMPL 2D keypoints to a converted sequence pkl.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--smpl_pkl",       required=True,
                   help="Conversion output pkl (has smpl_parameters dict).")
    p.add_argument("--smpl_model_path", required=True)
    p.add_argument("--out_pkl",        required=True)
    p.add_argument("--mhr_seq_pkl",    default="",
                   help="Optional: sequence_mhr.pkl (adds pred_keypoints_2d/3d).")
    p.add_argument("--device",         default="cpu")
    p.add_argument("--gender",         default="neutral")
    p.add_argument("--cx",             type=float, default=0.0)
    p.add_argument("--cy",             type=float, default=0.0)
    args = p.parse_args()

    device = torch.device(args.device)

    print(f"[INFO] Loading {args.smpl_pkl}")
    data = load_pkl(args.smpl_pkl)

    sp = data.get("smpl_parameters", {})
    T  = np.asarray(sp.get("global_orient", sp.get("root_orient"))).shape[0]
    print(f"[INFO] T={T} frames")

    # Focal length: stored in smpl_parameters or top-level
    focal_raw = sp.get("focal_length") if sp.get("focal_length") is not None else data.get("focal_length")
    if focal_raw is None:
        print("[WARN] No focal_length found; defaulting to 1000 px")
        focal = np.full(T, 1000.0, dtype=np.float32)
    else:
        focal = np.asarray(focal_raw, dtype=np.float32).reshape(-1)
        if focal.shape[0] == 1:
            focal = np.full(T, focal[0], dtype=np.float32)

    # SMPL forward pass
    print("[INFO] Loading SMPL model…")
    smpl_model = smplx.create(
        args.smpl_model_path, model_type="smpl",
        gender=args.gender, num_betas=10,
    ).to(device)
    smpl_model.eval()

    print("[INFO] Running SMPL forward pass…")
    joints_3d = run_smpl_forward(sp, smpl_model, device)  # (T, 24, 3)
    print(f"[INFO] joints_3d: {joints_3d.shape}")

    kpts2d_24 = project_joints_2d(joints_3d, focal, args.cx, args.cy)  # (T, 24, 2)

    # SMPL 21-joint subset (drop 0=pelvis, 22=L_hand, 23=R_hand — NRDF convention)
    keep21 = np.array([i for i in range(24) if i not in {0, 22, 23}], dtype=np.int64)
    kpts2d_21 = kpts2d_24[:, keep21, :]  # (T, 21, 2)

    print(f"[INFO] smpl_kpts2d_24: {kpts2d_24.shape}  range u=[{kpts2d_24[...,0].min():.0f},{kpts2d_24[...,0].max():.0f}]")

    # Inject into output
    data["smpl_kpts2d_24"] = kpts2d_24
    data["smpl_kpts2d_21"] = kpts2d_21
    # Store 3D joints too (useful for jitter analysis without needing SMPL re-run)
    data["smpl_joints_3d_24"] = joints_3d

    # Camera intrinsics used
    data["camera_intrinsics_used"] = {
        "model":            "pinhole",
        "fx_eq_fy":         True,
        "focal_length_px":  focal,
        "cx":               args.cx,
        "cy":               args.cy,
        "note":             ("cx/cy from --cx/--cy args; default 0 means coords "
                             "relative to principal point"),
    }

    # Optionally merge MHR 2D/3D kpts from sequence_mhr.pkl
    if args.mhr_seq_pkl and os.path.isfile(args.mhr_seq_pkl):
        print(f"[INFO] Merging MHR kpts from {args.mhr_seq_pkl}")
        mhr_seq = load_pkl(args.mhr_seq_pkl)
        for key in ("pred_keypoints_2d", "pred_keypoints_3d",
                    "pred_cam_t", "bbox", "frame_ids"):
            if key in mhr_seq:
                data[key] = mhr_seq[key]
        data["mhr_parameters"] = mhr_seq.get("mhr_parameters", {})
        print(f"  Added: pred_keypoints_2d {mhr_seq.get('pred_keypoints_2d', np.array([])).shape}")
        print(f"  Added: pred_keypoints_3d {mhr_seq.get('pred_keypoints_3d', np.array([])).shape}")

    save_pkl(data, args.out_pkl)
    print(f"\n[saved] {args.out_pkl}")
    print(f"  smpl_kpts2d_24 : {kpts2d_24.shape}")
    print(f"  smpl_kpts2d_21 : {kpts2d_21.shape}")
    print(f"  smpl_joints_3d : {joints_3d.shape}")
    if args.mhr_seq_pkl:
        print(f"  pred_keypoints_2d merged from MHR sequence")


if __name__ == "__main__":
    main()
