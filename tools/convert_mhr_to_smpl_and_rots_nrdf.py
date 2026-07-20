#!/usr/bin/env python3
"""
convert_mhr_to_smpl_and_rots_nrdf.py

Convert per-frame MHR/SAM3D pkls -> SMPL parameters, including:
- SMPL global/local joint rotations (body chain): (T, J, 3, 3)
- Angular velocities / accelerations in so(3)
- NRDF-compatible 21-joint subset (drops pelvis, hands: joints 0, 22, 23)

Must run with pixi (MHR env) from $MHR_DIR/tools/mhr_smpl_conversion/ so that
`from conversion import Conversion` resolves to the local conversion.py there.
The shell script handles the cd + pixi call automatically.

Usage:
  cd $MHR_DIR/tools/mhr_smpl_conversion
  pixi run python /path/to/tools/convert_mhr_to_smpl_and_rots_nrdf.py \\
      --in_dir  /path/to/extracted_frames \\
      --out_pkl /path/to/smpl_raw.pkl \\
      --smpl_model_path $SMPL_MODEL \\
      --mhr_assets $MHR_DIR/assets \\
      --gender neutral --device cpu --method pytorch \\
      --single_identity 1 --keep_src 0 --return_errors 1
"""

import os
import glob
import pickle
import argparse
from typing import Dict, Any, List, Tuple
from pathlib import Path

import numpy as np
import torch
import smplx
from scipy.spatial.transform import Rotation as SciR
from tqdm import tqdm

from mhr.mhr import MHR
from conversion import Conversion


# ── IO helpers ────────────────────────────────────────────────────────────────

def load_pkl(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)

def save_pkl(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def natural_key(s: str) -> List[Any]:
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", os.path.basename(s))]

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ── Frame data builders ───────────────────────────────────────────────────────

def build_mhr_vertices_from_frames(frames_src, device: str, key="pred_vertices"):
    verts = []
    for fr in frames_src:
        if key not in fr:
            raise KeyError(f"Missing '{key}' in frame dict. Available: {list(fr.keys())}")
        v = np.asarray(fr[key], dtype=np.float32)   # (V, 3)
        # SAM3D outputs meters; MHR conversion expects centimeters
        verts.append(torch.from_numpy(v * 100.0))
    return torch.stack(verts, dim=0).to(device=device, dtype=torch.float32)  # (T, V, 3)

def build_transl_from_frames(frames_src, device: torch.device, key="pred_cam_t"):
    ts = []
    for fr in frames_src:
        if key not in fr or fr[key] is None:
            raise KeyError(f"Missing '{key}' in frame dict. Available: {list(fr.keys())}")
        ts.append(torch.from_numpy(np.asarray(fr[key], dtype=np.float32).reshape(3,)))
    return torch.stack(ts, dim=0).to(device=device, dtype=torch.float32)  # (T, 3)

def build_focal_length_from_frames(frames_src, key="focal_length", device=torch.device("cpu")):
    focals = []
    for fr in frames_src:
        if key not in fr or fr[key] is None:
            raise KeyError(f"Missing '{key}' in frame dict. Available: {list(fr.keys())}")
        f = np.asarray(fr[key], dtype=np.float32).reshape(-1)
        focals.append(float(f[0]) if f.size == 1 else float(0.5 * (f[0] + f[1])))
    return torch.tensor(focals, device=device, dtype=torch.float32)  # (T,)


# ── Rotation math ─────────────────────────────────────────────────────────────

def axis_angle_to_rotmat_batch(aa: np.ndarray) -> np.ndarray:
    aa = np.asarray(aa, dtype=np.float64)
    if aa.ndim == 2:
        return SciR.from_rotvec(aa).as_matrix().astype(np.float32)
    T, J, _ = aa.shape
    return SciR.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(T, J, 3, 3).astype(np.float32)

def global_to_local_rotations(global_R: np.ndarray, parents: np.ndarray) -> np.ndarray:
    local_R = np.zeros_like(global_R)
    for j in range(global_R.shape[1]):
        p = int(parents[j])
        if p < 0 or p == j:
            local_R[:, j] = global_R[:, j]
        else:
            local_R[:, j] = np.transpose(global_R[:, p], (0, 2, 1)) @ global_R[:, j]
    return local_R

def matrix_log_map(Rm: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(Rm)
    wx, wy, wz = SciR.from_matrix(U @ Vt).as_rotvec()
    return np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]], dtype=np.float32)

def compute_angular_velocities(local_R: np.ndarray) -> np.ndarray:
    T, J = local_R.shape[:2]
    out = np.zeros((T - 1, J, 3, 3), dtype=np.float32)
    for t in tqdm(range(T - 1), desc="omega (log map)"):
        for j in range(J):
            out[t, j] = matrix_log_map(local_R[t, j].T @ local_R[t + 1, j])
    return out


# ── SMPL chain ────────────────────────────────────────────────────────────────

def extract_axis_angles(smpl_params: Dict[str, Any], B: int) -> Tuple[np.ndarray, np.ndarray]:
    def pick(*keys):
        for k in keys:
            if k in smpl_params and smpl_params[k] is not None:
                return to_numpy(smpl_params[k])
        return None
    go = pick("global_orient", "root_orient", "global_rot")
    bp = pick("body_pose", "pose_body", "body_pose_params")
    if go is None or bp is None:
        raise KeyError(f"Missing SMPL keys. Found: {list(smpl_params.keys())}")
    go = np.asarray(go, dtype=np.float32).reshape(B, 3)
    bp = np.asarray(bp, dtype=np.float32).reshape(B, -1)
    return go, bp.reshape(B, bp.shape[1] // 3, 3)

def compute_body_chain_global_rots(model, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    B = next(
        (int(v.shape[0]) for v in params.values()
         if isinstance(v, (torch.Tensor, np.ndarray)) and hasattr(v, "ndim") and v.ndim >= 1),
        None,
    )
    if B is None:
        raise ValueError("Could not infer batch size from params.")
    parents_full = to_numpy(model.parents).astype(np.int32)
    go_aa, bp_aa = extract_axis_angles(params, B)
    R_root = axis_angle_to_rotmat_batch(go_aa.reshape(B, 1, 3))[:, 0]  # (B, 3, 3)
    R_body = axis_angle_to_rotmat_batch(bp_aa)                          # (B, K, 3, 3)
    J = 1 + R_body.shape[1]
    parents = parents_full[:J].astype(np.int32)
    global_R = np.zeros((B, J, 3, 3), dtype=np.float32)
    global_R[:, 0] = R_root
    for j in range(1, J):
        p = int(parents[j])
        global_R[:, j] = (np.eye(3)[None] if p < 0 else global_R[:, p]) @ R_body[:, j - 1]
    return global_R, parents

def smpl_params_to_aa24(smpl_params: Dict[str, Any]) -> np.ndarray:
    go = to_numpy(smpl_params["global_orient"]).astype(np.float32).reshape(-1, 3)
    bp = to_numpy(smpl_params["body_pose"]).astype(np.float32).reshape(-1, 23, 3)
    return np.concatenate([go[:, None, :], bp], axis=1)  # (T, 24, 3)

def nrdf_select_21_from_smpl24(global_R, local_R, omega, alpha) -> Dict[str, np.ndarray]:
    keep = np.array([i for i in range(24) if i not in {0, 22, 23}], dtype=np.int64)
    return {
        "nrdf_keep_indices":   keep,
        "nrdf_global_rots_21": global_R[:, keep],
        "nrdf_local_rots_21":  local_R[:, keep],
        "nrdf_omega_21":       omega[:, keep],
        "nrdf_alpha_21":       alpha[:, keep],
    }

def nrdf_drop_to_21_from_aa24(aa24: np.ndarray) -> Dict[str, np.ndarray]:
    keep = np.array([i for i in range(24) if i not in {0, 22, 23}], dtype=np.int64)
    aa21 = aa24[:, keep, :]
    return {"nrdf_keep_indices": keep, "nrdf_pose_aa_21": aa21,
            "nrdf_pose_aa_63": aa21.reshape(aa21.shape[0], -1)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--in_dir",          required=True, help="Per-frame pkl directory.")
    ap.add_argument("--out_pkl",         required=True, help="Output sequence pkl path.")
    ap.add_argument("--smpl_model_path", required=True, help="SMPL neutral model .pkl file.")
    ap.add_argument("--mhr_assets",      required=True, help="Path to MHR assets/ folder.")
    ap.add_argument("--gender",          default="neutral", choices=["neutral", "male", "female"])
    ap.add_argument("--device",          default="cpu",     choices=["cuda", "cpu"])
    ap.add_argument("--method",          default="pytorch", choices=["pytorch", "pymomentum"])
    ap.add_argument("--lod",             type=int, default=1)
    ap.add_argument("--single_identity", type=int, default=1)
    ap.add_argument("--keep_src",        type=int, default=0)
    ap.add_argument("--return_errors",   type=int, default=1)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available.")
    device = torch.device(args.device)
    print(f"Device: {device}")

    # 1. Load per-frame pkls
    pkl_files = sorted(glob.glob(os.path.join(args.in_dir, "*.pkl")), key=natural_key)
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files in {args.in_dir}")

    frames_src = []
    for p in pkl_files:
        obj = load_pkl(p)
        if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
            fr = obj[0]
        elif isinstance(obj, dict):
            fr = obj
        else:
            print(f"  [skip] unexpected pkl format: {p}")
            continue
        fr["_src_path"] = p
        frames_src.append(fr)

    T = len(frames_src)
    print(f"Loaded {T} frames from {args.in_dir}")
    print(f"Frame keys: {list(frames_src[0].keys())}")

    # 2. Init MHR + SMPL + converter
    print(f"Initializing MHR (lod={args.lod}, assets={args.mhr_assets})...")
    mhr_model = MHR.from_files(lod=args.lod, folder=Path(args.mhr_assets), device=device)

    print("Initializing SMPL...")
    try:
        smpl_model = smplx.create(
            model_path=args.smpl_model_path, model_type="smpl",
            gender=args.gender, num_betas=10, batch_size=T,
        ).to(device)
    except Exception as e:
        print(f"  [warn] smplx.create failed ({e}), falling back to smplx.SMPL()")
        smpl_model = smplx.SMPL(
            model_path=args.smpl_model_path, gender=args.gender,
            num_betas=10, batch_size=T,
        ).to(device)

    converter = Conversion(mhr_model=mhr_model, smpl_model=smpl_model, method=args.method)

    # 3. Build batched inputs
    mhr_vertices = build_mhr_vertices_from_frames(frames_src, device=device)
    mhr_transl   = build_transl_from_frames(frames_src, device=device)
    mhr_focal    = build_focal_length_from_frames(frames_src, device=device)

    # 4. MHR vertices → SMPL params
    print("Converting MHR vertices → SMPL parameters...")
    results = converter.convert_mhr2smpl(
        mhr_vertices=mhr_vertices,
        mhr_parameters=None,
        single_identity=bool(args.single_identity),
        return_smpl_parameters=True,
        return_smpl_vertices=False,
        return_fitting_errors=bool(args.return_errors),
        return_smpl_meshes=False,
    )

    smpl_params = getattr(results, "result_parameters", None)
    if not isinstance(smpl_params, dict):
        raise RuntimeError("Converter did not return a parameter dict.")

    transl_fit = smpl_params.get("transl", None)
    smpl_params["transl_fit"]   = transl_fit
    smpl_params["transl"]       = mhr_transl
    smpl_params["transl_final"] = mhr_transl + transl_fit if transl_fit is not None else mhr_transl
    smpl_params["focal_length"] = mhr_focal

    smpl_aa24 = smpl_params_to_aa24(smpl_params)   # (T, 24, 3)
    errors = getattr(results, "result_errors", None) if args.return_errors else None

    # 5. Global / local rotations + angular derivatives
    print("Computing body-chain rotations...")
    smpl_global_rots, smpl_parents = compute_body_chain_global_rots(smpl_model, smpl_params)
    smpl_local_rots = global_to_local_rotations(smpl_global_rots, smpl_parents)
    smpl_omega = compute_angular_velocities(smpl_local_rots)     # (T-1, J, 3, 3)
    smpl_alpha = smpl_omega[1:] - smpl_omega[:-1]               # (T-2, J, 3, 3)

    # 6. NRDF 21-joint subset
    nrdf_pack = None
    if smpl_global_rots.shape[1] == 24:
        nrdf_pack = nrdf_select_21_from_smpl24(
            smpl_global_rots, smpl_local_rots, smpl_omega, smpl_alpha)

    # 7. Save
    batched_smpl_params = {k: to_numpy(v) for k, v in smpl_params.items()}
    out_obj: Dict[str, Any] = {
        "meta": {
            "in_dir": args.in_dir, "n_frames": T,
            "smpl_model_path": args.smpl_model_path,
            "gender": args.gender, "method": args.method,
            "lod": args.lod, "single_identity": bool(args.single_identity),
        },
        "focal_length":               to_numpy(mhr_focal),
        "smpl_parameters":            batched_smpl_params,
        "smpl_joint_parents":         smpl_parents,
        "smpl_global_rots":           smpl_global_rots,
        "smpl_local_rots":            smpl_local_rots,
        "smpl_angular_velocities":    smpl_omega,
        "smpl_angular_accelerations": smpl_alpha,
        "smpl_axis_angle_24":         smpl_aa24,
        "smpl_axis_angle_24_flat":    smpl_aa24.reshape(T, -1),
        "fit_errors":                 to_numpy(errors) if errors is not None else None,
    }
    if nrdf_pack is not None:
        out_obj["nrdf"] = nrdf_pack
        out_obj["nrdf_aa"] = nrdf_drop_to_21_from_aa24(smpl_aa24)

    save_pkl(out_obj, args.out_pkl)
    print(f"\nSaved → {args.out_pkl}")
    print(f"  smpl_global_rots : {smpl_global_rots.shape}")
    print(f"  smpl_local_rots  : {smpl_local_rots.shape}")
    print(f"  omega            : {smpl_omega.shape}")
    print(f"  alpha            : {smpl_alpha.shape}")
    if nrdf_pack:
        print(f"  nrdf_global_21   : {nrdf_pack['nrdf_global_rots_21'].shape}")


if __name__ == "__main__":
    main()
