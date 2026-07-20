#!/usr/bin/env python3
"""
rollout_motion_optim.py

Physics-guided rollout-based global human motion reconstruction.

Core idea
---------
Instead of optimizing the full pose-state sequence {x_t, R_t} directly with
finite-difference regularizers bolted on, we optimize a *control sequence*
{a_t, α_t} and *integrate* it forward through discrete dynamics.

  Optimize:  x0, v0, ψ0, ω0,  {a_t}_{t=0..T-2},  {α_t}_{t=0..T-2}
             [optionally: one shared static camera R_cam, t_cam]

  Rollout (symplectic Euler):
    v_{t+1}  = v_t  + a_t  · Δt
    x_{t+1}  = x_t  + v_{t+1} · Δt          ← uses updated velocity
    ω_{t+1}  = ω_t  + α_t  · Δt
    ψ_{t+1}  = ψ_t  + ω_{t+1} · Δt

  Observation model:
    R_t      = R_yaw(ψ_t) ⊗ R_pitchroll_SAM3D_t   (yaw replaced, pitch/roll kept)
    J_t      = SMPL(θ_t^SAM, β^SAM, R_t, x_t)
    ûₜⱼ     = π(Jₜⱼ; K, R_cam, t_cam)

  Losses:
    L_reproj   — Huber reprojection on 2D joints
    L_contact  — foot penetration + floating penalty
    L_ctrl_lat — lateral (cross-track) acceleration suppression  ||a_t⊥||²
    L_smooth   — forward acceleration smoothness  ||Δ(a_t·d)||²
    L_yaw      — yaw-rate regularization  α_t²
    L_anchor   — weak L2 pull toward SAM3D/HS translation (prevents drift)

Optional Gaussian pulse model (--use_pulse_model)
-------------------------------------------------
Instead of a raw per-frame a_ctrl, parameterize the FORWARD acceleration as a
sum of K Gaussian pulses:

    a_fwd(t) = Σ_k  α_k · exp(-(t - μ_k)² / (2σ_k²))

  Each pulse = a physical motion event (push-off, burst, deceleration).
  Lateral acceleration remains a small per-frame parameter.

Usage
-----
python core/rollout_motion_optim.py \\
    --in_pkl  $SAM3D_DIR/outputs/smpl_sequences/<Athlete>/smpl_sequence.pkl \\
    --out_pkl outputs/.../rollout_result.pkl \\
    --smpl_model_path /path/to/SMPL_NEUTRAL.pkl \\
    --video   /path/to/video.mp4 \\
    --device  cuda \\
    --iters 1000 --stage1_iters 300
"""

import os, sys, argparse, pickle, copy, time, warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════════════════════
# Section 0: Shared utilities (I/O, rotation, camera, SMPL, projection, losses)
# ══════════════════════════════════════════════════════════════════════════════

def load_pkl(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def ensure_T(arr: Any, T: int, name: str) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32).reshape(-1)
    if a.size == 1 and T > 1:
        a = np.repeat(a, T)
    if a.size != T:
        raise ValueError(f"{name}: expected length {T}, got {a.shape}")
    return a


def ensure_TxD(arr: Any, T: int, D: int, name: str) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1 and a.size == D:
        a = a[None, :]
    if a.ndim == 2 and a.shape[0] == 1 and T > 1:
        a = np.repeat(a, T, axis=0)
    if a.shape != (T, D):
        raise ValueError(f"{name}: expected ({T},{D}), got {a.shape}")
    return a


def to_torch(x: Any, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(np.asarray(x, dtype=np.float32), device=device, dtype=dtype)


def get_video_info(video_path: str) -> Tuple[float, float, float, int, int]:
    """Returns (cx, cy, fps, width, height) from video file via OpenCV."""
    if cv2 is None:
        return 0.0, 0.0, 30.0, 0, 0
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        warnings.warn(f"Cannot open video: {video_path}")
        return 0.0, 0.0, 30.0, 0, 0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    cap.release()
    return W / 2.0, H / 2.0, float(fps), W, H


def rotvec_to_matrix(rv: torch.Tensor) -> torch.Tensor:
    """Batch Rodrigues: (..., 3) axis-angle → (..., 3, 3) rotation matrix."""
    shape = rv.shape[:-1]
    rv_flat = rv.reshape(-1, 3)
    theta = torch.norm(rv_flat, dim=-1, keepdim=True).clamp(min=1e-8)
    k = rv_flat / theta
    zeros = torch.zeros(rv_flat.shape[0], device=rv.device, dtype=rv.dtype)
    K = torch.stack(
        [zeros, -k[:, 2],  k[:, 1],
          k[:, 2],  zeros, -k[:, 0],
         -k[:, 1],  k[:, 0],  zeros],
        dim=-1,
    ).reshape(-1, 3, 3)
    I3 = torch.eye(3, device=rv.device, dtype=rv.dtype).unsqueeze(0)
    s, c = torch.sin(theta).unsqueeze(-1), (1.0 - torch.cos(theta)).unsqueeze(-1)
    return (I3 + s * K + c * (K @ K)).reshape(*shape, 3, 3)


def matrix_to_rotvec_np(R: np.ndarray) -> np.ndarray:
    if not _SCIPY_OK:
        raise RuntimeError("scipy required for matrix_to_rotvec_np")
    shape = R.shape[:-2]
    return ScipyRotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(*shape, 3).astype(np.float32)


def rotvec_to_matrix_np(rv: np.ndarray) -> np.ndarray:
    if not _SCIPY_OK:
        raise RuntimeError("scipy required for rotvec_to_matrix_np")
    shape = rv.shape[:-1]
    return ScipyRotation.from_rotvec(rv.reshape(-1, 3)).as_matrix().reshape(*shape, 3, 3).astype(np.float32)


def load_sam3d_camera_poses(
    data: Dict[str, Any],
    cam_R_key: str = "cam_R",
    cam_t_key: str = "cam_t",
    cam_pkl_path: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load SAM3D camera (R, t) from several fallback locations in the data dict."""
    def _parse_R(arr):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 9:
            arr = arr.reshape(-1, 3, 3)
        if arr.ndim != 3 or arr.shape[1:] != (3, 3):
            raise ValueError(f"Unexpected cam_R shape: {arr.shape}")
        return arr

    def _parse_t(arr):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None] if arr.size == 3 else arr.reshape(-1, 3)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Unexpected cam_t shape: {arr.shape}")
        return arr

    if cam_pkl_path is not None and os.path.isfile(cam_pkl_path):
        try:
            d = load_pkl(cam_pkl_path)
            R_raw = d.get("cam_R") or d.get("R")
            t_raw = d.get("cam_t") or d.get("t")
            if R_raw is not None and t_raw is not None:
                return _parse_R(np.asarray(R_raw)), _parse_t(np.asarray(t_raw))
        except Exception as e:
            warnings.warn(f"Failed to load cam_pkl {cam_pkl_path}: {e}")

    sp = data.get("smpl_parameters", {})
    candidates = [
        (f"smpl_parameters['{cam_R_key}']", sp.get(cam_R_key),              sp.get(cam_t_key)),
        ("data['camera']",                  data.get("camera", {}).get("R"), data.get("camera", {}).get("t")),
        (f"data['{cam_R_key}']",             data.get(cam_R_key),             data.get(cam_t_key)),
        ("data['cam_R'/'cam_t']",            data.get("cam_R"),               data.get("cam_t")),
    ]
    for src, R_raw, t_raw in candidates:
        if R_raw is not None and t_raw is not None:
            try:
                print(f"[INFO] Loaded SAM3D cam poses from {src}")
                return _parse_R(np.asarray(R_raw)), _parse_t(np.asarray(t_raw))
            except Exception as e:
                warnings.warn(f"Parse error {src}: {e}")

    print("[WARNING] No SAM3D camera poses found — assuming static camera (identity).")
    return None, None


def build_world_frame_cameras(cam_R: np.ndarray, cam_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize SAM3D camera poses to the first frame's coordinate system."""
    R0T = cam_R[0].T
    R_rel = cam_R @ R0T[None]
    t_rel = cam_t - (R_rel @ cam_t[0][:, None]).squeeze(-1)
    return R_rel.astype(np.float32), t_rel.astype(np.float32)


def run_smpl(
    smpl_model,
    global_orient: torch.Tensor,
    body_pose: torch.Tensor,
    betas: torch.Tensor,
    transl: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """SMPL forward pass → (T, 24, 3) world-frame joints."""
    out = smpl_model(
        global_orient=global_orient, body_pose=body_pose,
        betas=betas, transl=transl, return_verts=False, pose2rot=True,
    )
    return out.joints[:, :24, :]


def run_smpl_zero_transl(smpl_model, global_orient, body_pose, betas, device):
    T = global_orient.shape[0]
    return run_smpl(smpl_model, global_orient, body_pose, betas,
                    torch.zeros(T, 3, device=device), device)


def project_joints(
    J_world: torch.Tensor,
    R_cam: torch.Tensor,
    t_cam: torch.Tensor,
    focal: torch.Tensor,
    cx: float,
    cy: float,
) -> torch.Tensor:
    """Project (T, J, 3) world joints to (T, J, 2) image pixels."""
    T, nJ, _ = J_world.shape
    J_cam = (R_cam.unsqueeze(1) @ J_world.unsqueeze(-1)).squeeze(-1) + t_cam.unsqueeze(1)
    Zd = J_cam[..., 2:3].clamp(min=1e-3)
    u = focal.view(T, 1, 1) * J_cam[..., 0:1] / Zd + cx
    v = focal.view(T, 1, 1) * J_cam[..., 1:2] / Zd + cy
    return torch.cat([u, v], dim=-1)


def loss_reprojection(
    uv_pred: torch.Tensor,
    uv_obs: torch.Tensor,
    huber_delta: float = 15.0,
    frame_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    diff = uv_pred - uv_obs
    h = F.huber_loss(diff, torch.zeros_like(diff), delta=huber_delta, reduction="none")
    if frame_weights is not None:
        h = h * frame_weights[:, None, None]
    return h.mean()


_FOOT_JOINT_IDS = [7, 8, 10, 11]   # SMPL-24: left/right ankle + foot toe


def loss_ground_contact(
    J_world: torch.Tensor,
    floor_y: torch.Tensor,
    foot_ids: List[int] = _FOOT_JOINT_IDS,
    margin: float = 0.05,
    w_penetration: float = 1.0,
    w_liftoff: float = 0.5,
) -> torch.Tensor:
    """Foot penetration + floating penalty against a learnable floor plane."""
    T = J_world.shape[0]
    foot_y = J_world[:, foot_ids, 1]
    fy = floor_y.expand(T) if floor_y.dim() == 0 else floor_y
    L_pen  = (torch.relu(fy.unsqueeze(1) - foot_y) ** 2).mean()
    L_lift = (torch.relu(foot_y.min(dim=1).values - fy - margin) ** 2).mean()
    return w_penetration * L_pen + w_liftoff * L_lift


def estimate_sprint_direction(transl_np: np.ndarray, plane: str = "xz") -> np.ndarray:
    """PCA-based sprint direction from the XZ trajectory (Y = up)."""
    if plane == "xz":
        coords = transl_np[:, [0, 2]].astype(np.float64)
        coords -= coords.mean(0)
        if len(coords) >= 2:
            cov = coords.T @ coords / max(len(coords) - 1, 1)
            sprint_2d = np.linalg.eigh(cov)[1][:, -1]
        else:
            sprint_2d = np.diff(transl_np[:, [0, 2]], axis=0).mean(0)
        nrm = np.linalg.norm(sprint_2d)
        sprint_2d = sprint_2d / nrm if nrm > 1e-6 else np.array([0.0, 1.0])
        d = np.array([sprint_2d[0], 0.0, sprint_2d[1]], dtype=np.float32)
    else:
        coords = transl_np.astype(np.float64) - transl_np.mean(0)
        cov = coords.T @ coords / max(len(coords) - 1, 1)
        d = np.linalg.eigh(cov)[1][:, -1].astype(np.float32)
    d /= np.linalg.norm(d) + 1e-8
    net = transl_np[-1] - transl_np[0]
    net_proj = np.array([net[0], 0.0, net[2]]) if plane == "xz" else net
    if np.dot(d, net_proj.astype(np.float32)) < 0:
        d = -d
    print(f"[INFO] Sprint direction (plane={plane}): d={d.round(4)}")
    return d


def init_human_world_pose(
    p_cam_t: np.ndarray,
    R_cam0: np.ndarray,
    t_cam0: np.ndarray,
    global_orient_cam: np.ndarray,
    smpl_model,
    body_pose_np: np.ndarray,
    betas_np: np.ndarray,
    pelvis_id: int = 0,
    device: torch.device = torch.device("cpu"),
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert camera-frame SMPL pose → world-frame (transl_world_init, go_world_init)."""
    T = p_cam_t.shape[0]
    R_cam0T = R_cam0.transpose(0, 2, 1)
    p_world_t = (R_cam0T @ (p_cam_t - t_cam0)[..., None]).squeeze(-1)
    with torch.no_grad():
        J_zero = run_smpl_zero_transl(
            smpl_model, to_torch(global_orient_cam, device),
            to_torch(body_pose_np, device), to_torch(betas_np, device), device,
        )
    pelvis_offset = J_zero[:, pelvis_id, :].cpu().numpy()
    transl_world_init = p_world_t - pelvis_offset
    R_go_cam      = rotvec_to_matrix_np(global_orient_cam)
    go_world_init = matrix_to_rotvec_np(R_cam0T @ R_go_cam)
    return transl_world_init.astype(np.float32), go_world_init.astype(np.float32)

try:
    import smplx
except ImportError:
    smplx = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from scipy.spatial.transform import Rotation as ScipyRotation
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

try:
    from scipy.signal import medfilt as _medfilt
    _SCIPY_SIGNAL_OK = True
except ImportError:
    _SCIPY_SIGNAL_OK = False

# ── Biomechanics constants ────────────────────────────────────────────────────

# Key SMPL-24 joints for local rotation optimization (hips, spine, knees, shoulders)
_KEY_JOINT_SMPL = [1, 2, 3, 4, 5, 6, 9, 16, 17]      # 9 joints (sorted)
_KEY_BP_STARTS  = [(j - 1) * 3 for j in _KEY_JOINT_SMPL]  # [0,3,6,9,12,15,24,45,48]
N_KEY_JOINTS    = len(_KEY_JOINT_SMPL)

# Approximate segment mass fractions (Winter 1990) mapped to SMPL-24 joints.
# Represents relative inertia contribution of each joint for WBAM computation.
_SMPL24_SEG_MASSES_NP = np.array([
    0.142,  #  0 pelvis
    0.100,  #  1 left_hip → left thigh
    0.100,  #  2 right_hip → right thigh
    0.050,  #  3 spine1
    0.046,  #  4 left_knee → left shank
    0.046,  #  5 right_knee → right shank
    0.050,  #  6 spine2
    0.014,  #  7 left_ankle → left foot
    0.014,  #  8 right_ankle → right foot
    0.070,  #  9 spine3
    0.003,  # 10 left_foot_toe
    0.003,  # 11 right_foot_toe
    0.081,  # 12 neck → head
    0.028,  # 13 left_collar → left upper arm
    0.028,  # 14 right_collar → right upper arm
    0.010,  # 15 head
    0.016,  # 16 left_shoulder → left forearm
    0.016,  # 17 right_shoulder → right forearm
    0.006,  # 18 left_elbow → left hand
    0.006,  # 19 right_elbow → right hand
    0.003,  # 20 left_wrist
    0.003,  # 21 right_wrist
    0.001,  # 22 left_hand_tip
    0.001,  # 23 right_hand_tip
], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Section 1b: Confidence weighting utilities
# ══════════════════════════════════════════════════════════════════════════════

def compute_conf_weights(
    fit_errors: np.ndarray,
    sigma: float = 1.0,
    min_weight: float = 0.1,
) -> np.ndarray:
    """
    Convert per-frame SMPL fit errors to normalized confidence weights.

    Frames where SAM3D fitted poorly (high fit_error) get a lower weight in
    both the reprojection loss and the anchor losses, so noisy pose estimates
    do not dominate the optimization.

    Weight formula:
        w_t = clip(exp(-fit_errors_t / (sigma * median(fit_errors))), min_weight, 1.0)
    then normalized so mean(w) = 1.0 to keep the overall loss magnitude stable
    (existing weight values such as w_reproj, w_anchor stay meaningful).

    Parameters
    ----------
    fit_errors : (T,) per-frame SMPL fitting error (from 'fit_errors' in pkl)
    sigma      : softness of the exponential curve; larger = gentler contrast.
                 sigma=1.0 → frames at 2× the median get weight ≈ 0.37.
    min_weight : floor to avoid zeroing any frame entirely.

    Returns
    -------
    weights : (T,) float32, normalized so mean = 1.0
    """
    fe = np.asarray(fit_errors, dtype=np.float32)
    med = float(np.median(fe)) + 1e-6
    w = np.exp(-(fe / (sigma * med)))
    w = np.clip(w, min_weight, 1.0)
    w = w / (w.mean() + 1e-8)
    return w


# ══════════════════════════════════════════════════════════════════════════════
# Section 2: Rollout dynamics
# ══════════════════════════════════════════════════════════════════════════════

def rollout_states(
    x0: torch.Tensor,
    v0: torch.Tensor,
    yaw0: torch.Tensor,
    yaw_rate0: torch.Tensor,
    a_ctrl: torch.Tensor,
    yaw_acc_ctrl: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Integrate the discrete dynamics forward from (x0, v0, ψ0, ω0) using
    T-1 control steps (a_ctrl, yaw_acc_ctrl).

    Symplectic Euler integration (velocity updated before position):
        v_{t+1}  = v_t  + a_t  · dt
        x_{t+1}  = x_t  + v_{t+1} · dt   ← uses v_{t+1}, not v_t

    This conserves energy better than standard Euler and is more stable
    for long rollouts with acceleration-level controls.

    Parameters
    ----------
    x0, v0         : (3,)    initial translation and velocity
    yaw0, yaw_rate0: scalar  initial yaw angle and angular velocity
    a_ctrl         : (T-1, 3) translational acceleration control sequence
    yaw_acc_ctrl   : (T-1,)  yaw angular acceleration control sequence
    dt             : float    time step (1/fps)

    Returns
    -------
    x_seq      : (T, 3)   position trajectory
    v_seq      : (T, 3)   velocity trajectory
    yaw_seq    : (T,)     yaw angle trajectory
    yaw_rate_seq: (T,)    yaw angular velocity trajectory
    """
    T = a_ctrl.shape[0] + 1

    x_list        = [x0]
    v_list        = [v0]
    yaw_list      = [yaw0.unsqueeze(0) if yaw0.dim() == 0 else yaw0]
    yaw_rate_list = [yaw_rate0.unsqueeze(0) if yaw_rate0.dim() == 0 else yaw_rate0]

    for t in range(T - 1):
        # Symplectic Euler: update velocity first, then position
        v_next        = v_list[t] + a_ctrl[t] * dt            # (3,)
        x_next        = x_list[t] + v_next * dt                # (3,)
        yaw_rate_next = yaw_rate_list[t] + yaw_acc_ctrl[t:t+1] * dt  # (1,)
        yaw_next      = yaw_list[t] + yaw_rate_next * dt       # (1,)

        x_list.append(x_next)
        v_list.append(v_next)
        yaw_list.append(yaw_next)
        yaw_rate_list.append(yaw_rate_next)

    x_seq         = torch.stack(x_list,        dim=0)   # (T, 3)
    v_seq         = torch.stack(v_list,        dim=0)   # (T, 3)
    yaw_seq       = torch.cat(yaw_list,        dim=0)   # (T,)
    yaw_rate_seq  = torch.cat(yaw_rate_list,   dim=0)   # (T,)

    return x_seq, v_seq, yaw_seq, yaw_rate_seq


# ══════════════════════════════════════════════════════════════════════════════
# Section 3: Orientation helpers
# ══════════════════════════════════════════════════════════════════════════════

def yaw_to_rotmat(yaw: torch.Tensor) -> torch.Tensor:
    """
    Convert a sequence of yaw angles to rotation matrices around the Y axis.

    Convention (Y-up world frame):
        R_y(ψ) = [[cos ψ,  0,  sin ψ],
                  [0,      1,  0    ],
                  [-sin ψ, 0,  cos ψ]]

    Parameters
    ----------
    yaw : (T,)  yaw angles in radians

    Returns
    -------
    R : (T, 3, 3)
    """
    T   = yaw.shape[0]
    c   = torch.cos(yaw)          # (T,)
    s   = torch.sin(yaw)          # (T,)
    z   = torch.zeros_like(yaw)
    o   = torch.ones_like(yaw)

    R = torch.stack([
        c,  z,  s,
        z,  o,  z,
       -s,  z,  c,
    ], dim=-1).reshape(T, 3, 3)
    return R


def extract_yaw_from_rotmat(R: torch.Tensor) -> torch.Tensor:
    """
    Extract yaw angle (rotation around Y) from rotation matrices.

    Project the body's canonical forward vector [0, 0, 1] through R to
    get the world-frame facing direction, then compute yaw from its XZ projection.

    Parameters
    ----------
    R : (T, 3, 3)

    Returns
    -------
    yaw : (T,)  yaw angles in radians
    """
    # R @ [0,0,1]^T = third column of R
    fwd_x = R[:, 0, 2]   # x component of forward direction
    fwd_z = R[:, 2, 2]   # z component
    return torch.atan2(fwd_x, fwd_z)


def extract_yaw_from_rotmat_np(R: np.ndarray) -> np.ndarray:
    """NumPy version for initialization."""
    fwd_x = R[:, 0, 2]
    fwd_z = R[:, 2, 2]
    return np.arctan2(fwd_x, fwd_z).astype(np.float32)


def combine_yaw_with_sam3d(
    yaw_new: torch.Tensor,
    R_sam: torch.Tensor,
) -> torch.Tensor:
    """
    Replace the yaw component in SAM3D rotations while preserving pitch and roll.

    Decomposition:
        R_sam   = R_yaw_sam · R_pitchroll
        R_new   = R_yaw_new · R_pitchroll
                = R_yaw_new · R_yaw_sam^T · R_sam

    This ensures the body's lean and tilt (from SAM3D, which come from
    the articulated pose) are preserved while only the global heading is replaced.

    Parameters
    ----------
    yaw_new : (T,)      rollout yaw angles
    R_sam   : (T, 3, 3) SAM3D world-frame root rotation matrices

    Returns
    -------
    R_combined : (T, 3, 3)
    """
    R_yaw_sam = yaw_to_rotmat(extract_yaw_from_rotmat(R_sam))   # (T, 3, 3)
    R_pitchroll = R_yaw_sam.transpose(-1, -2) @ R_sam            # (T, 3, 3)
    R_yaw_new   = yaw_to_rotmat(yaw_new)                         # (T, 3, 3)
    return R_yaw_new @ R_pitchroll                                # (T, 3, 3)


def rotmat_to_rotvec_diff(R: torch.Tensor) -> torch.Tensor:
    """
    Differentiable rotation matrix → axis-angle conversion.

    Uses the Rodrigues formula:
        angle  = arccos((trace(R) - 1) / 2)
        axis   = skew(R) / (2 sin(angle))
        rotvec = angle · axis

    Numerically stable for angle ∈ (0, π).
    For angle ≈ 0 (identity), returns zero vector.

    Parameters
    ----------
    R : (T, 3, 3)

    Returns
    -------
    rv : (T, 3)
    """
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]                 # (T,)
    cos_a = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    angle = torch.acos(cos_a)                                      # (T,)

    # Skew-symmetric part gives 2 sin(angle) * axis
    skew = torch.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1],
    ], dim=-1)                                                     # (T, 3)

    safe_sin = (2.0 * torch.sin(angle)).clamp(min=1e-7)
    rotvec = skew * (angle / safe_sin).unsqueeze(-1)               # (T, 3)

    # At angle ≈ 0 the formula is 0/0; the correct limit is rotvec = 0
    near_zero = (angle < 1e-6).unsqueeze(-1)
    return torch.where(near_zero, torch.zeros_like(rotvec), rotvec)


# ══════════════════════════════════════════════════════════════════════════════
# Section 3b: Full-DOF orientation rollout (SO(3) integration)
# ══════════════════════════════════════════════════════════════════════════════

def exp_so3_batch(v: torch.Tensor) -> torch.Tensor:
    """
    Differentiable exponential map on SO(3) using the Rodrigues formula.

    R = cos(θ)·I + sin(θ)/θ·[v]× + (1−cos(θ))/θ²·(v⊗v)

    Parameters
    ----------
    v : (..., 3)  axis-angle vectors  (angle = ||v||)

    Returns
    -------
    R : (..., 3, 3)  rotation matrices
    """
    shape = v.shape[:-1]
    v_flat = v.reshape(-1, 3)
    N = v_flat.shape[0]

    theta = v_flat.norm(dim=-1)                            # (N,)
    safe_theta = theta.clamp(min=1e-7)
    k = v_flat / safe_theta.unsqueeze(-1)                  # (N, 3) unit axis

    c  = torch.cos(theta)                                  # (N,)
    s  = torch.sin(theta)                                  # (N,)
    t_ = 1.0 - c                                           # (N,)
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]

    R_flat = torch.stack([
        t_*kx*kx + c,    t_*kx*ky - s*kz, t_*kx*kz + s*ky,
        t_*kx*ky + s*kz, t_*ky*ky + c,    t_*ky*kz - s*kx,
        t_*kx*kz - s*ky, t_*ky*kz + s*kx, t_*kz*kz + c,
    ], dim=-1).reshape(N, 3, 3)

    I = torch.eye(3, device=v.device, dtype=v.dtype).unsqueeze(0).expand(N, 3, 3)
    near_zero = (theta < 1e-7).reshape(N, 1, 1)
    R_flat = torch.where(near_zero, I, R_flat)
    return R_flat.reshape(*shape, 3, 3)


def rollout_orientation_full_dof(
    R0: torch.Tensor,
    w3d_0: torch.Tensor,
    alpha3d: torch.Tensor,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Integrate full 3D orientation on SO(3) using symplectic Euler.

    ω_{t+1} = ω_t + α_t · dt                    (angular velocity update)
    R_{t+1} = R_t @ exp_SO3(ω_{t+1} · dt)       (body-frame rotation step)

    Parameters
    ----------
    R0      : (3, 3)    initial rotation matrix (from SAM3D at t=0)
    w3d_0   : (3,)      initial 3D angular velocity [rad/s]
    alpha3d : (T-1, 3)  3D angular acceleration controls [rad/s²]
    dt      : float     time step

    Returns
    -------
    R_seq : (T, 3, 3)   rotation matrix sequence
    w_seq : (T, 3)      angular velocity sequence
    """
    T = alpha3d.shape[0] + 1
    R_cur = R0          # (3, 3)
    w_cur = w3d_0       # (3,)
    R_list = [R_cur.unsqueeze(0)]
    w_list = [w_cur.unsqueeze(0)]

    for t in range(T - 1):
        w_next   = w_cur + alpha3d[t] * dt                   # (3,)
        delta_v  = (w_next * dt).unsqueeze(0)                # (1, 3)
        dR       = exp_so3_batch(delta_v).squeeze(0)         # (3, 3)
        R_next   = R_cur @ dR                                # (3, 3)
        R_list.append(R_next.unsqueeze(0))
        w_list.append(w_next.unsqueeze(0))
        R_cur = R_next
        w_cur = w_next

    return torch.cat(R_list, dim=0), torch.cat(w_list, dim=0)  # (T,3,3), (T,3)


def init_full_dof_orient_np(
    go_world_init: np.ndarray,
    dt: float,
    smooth_sigma: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Derive initial 3D angular velocity and angular acceleration controls from
    the SAM3D world-frame orientation sequence.

    Parameters
    ----------
    go_world_init : (T, 3)  axis-angle orientations in world frame
    dt            : float   time step
    smooth_sigma  : float   Gaussian smoothing for derivatives

    Returns
    -------
    w3d_0_init  : (3,)      initial 3D angular velocity [rad/s]
    alpha3d_init: (T-1, 3)  initial angular acceleration controls [rad/s²]
    """
    from scipy.spatial.transform import Rotation as ScR

    R = rotvec_to_matrix_np(go_world_init)                    # (T, 3, 3)
    # Relative rotation between consecutive frames
    dR = R[1:] @ R[:-1].transpose(0, 2, 1)                   # (T-1, 3, 3)
    # Log map → axis-angle → angular velocity
    w_seq = ScR.from_matrix(dR).as_rotvec().astype(np.float32) / dt  # (T-1, 3)

    if smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter1d as gf1d
            w_seq = gf1d(w_seq.astype(np.float64), sigma=smooth_sigma, axis=0).astype(np.float32)
        except ImportError:
            pass

    w3d_0_init = w_seq[0]                                     # (3,)

    # Angular acceleration via finite diff of angular velocity
    alpha = (w_seq[1:] - w_seq[:-1]) / dt                    # (T-2, 3)
    alpha3d_init = np.concatenate([alpha, alpha[-1:]], axis=0) # (T-1, 3)
    return w3d_0_init, alpha3d_init


# ── Local rotation rollout ────────────────────────────────────────────────────

def rollout_local_rot(
    theta0_key: torch.Tensor,
    omega0_key: torch.Tensor,
    alpha_key:  torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """
    Angular velocity rollout for local joint rotations (axis-angle space).

    Integrates each key joint independently using symplectic Euler:
        ω_{t+1,j} = ω_{t,j} + α_{t,j} · dt
        θ_{t+1,j} = θ_{t,j} + ω_{t+1,j} · dt

    This is a linear approximation on the SO(3) manifold — valid for the
    smooth, bounded joint motions in sprinting (no large-angle flips).

    theta0_key : (n_key, 3)    initial key joint axis-angles
    omega0_key : (n_key, 3)    initial angular velocities [rad/s]
    alpha_key  : (T-1, n_key, 3)  angular acceleration controls [rad/s²]
    dt         : float         time step

    Returns theta_seq : (T, n_key, 3)
    """
    T = alpha_key.shape[0] + 1
    theta_cur = theta0_key   # (n_key, 3)
    omega_cur = omega0_key   # (n_key, 3)
    theta_list = [theta_cur]

    for t in range(T - 1):
        omega_next = omega_cur + alpha_key[t] * dt
        theta_next = theta_cur + omega_next * dt
        theta_list.append(theta_next)
        theta_cur = theta_next
        omega_cur = omega_next

    return torch.stack(theta_list, dim=0)     # (T, n_key, 3)


def init_local_rot_np(
    body_pose_np: np.ndarray,
    dt: float,
    smooth_sigma: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Derive initial state and control for local rotation rollout from SAM3D body_pose.

    Parameters
    ----------
    body_pose_np : (T, 69) SAM3D body_pose axis-angles
    dt           : time step
    smooth_sigma : Gaussian smoothing frames

    Returns
    -------
    theta0_key   : (n_key, 3)    initial key joint angles (frame 0)
    omega0_key   : (n_key, 3)    initial angular velocities
    alpha_key_init: (T-1, n_key, 3) initial angular acceleration controls
    """
    try:
        from scipy.ndimage import gaussian_filter1d as gf1d
        _gf = lambda x: gf1d(x.astype(np.float64), sigma=smooth_sigma, axis=0).astype(np.float32)
    except ImportError:
        _gf = lambda x: x

    # Extract key joints: (T, n_key, 3)
    theta_key = np.stack([body_pose_np[:, bp_s:bp_s+3] for bp_s in _KEY_BP_STARTS], axis=1)

    theta_smooth = np.stack([_gf(theta_key[:, i, :]) for i in range(N_KEY_JOINTS)], axis=1)

    # Angular velocity: finite diff of smoothed theta
    omega = (theta_smooth[1:] - theta_smooth[:-1]) / dt      # (T-1, n_key, 3)
    omega0_key = omega[0]                                      # (n_key, 3)

    # Angular acceleration: second diff of smoothed theta
    alpha = (omega[1:] - omega[:-1]) / dt                     # (T-2, n_key, 3)
    alpha_key_init = np.concatenate([alpha, alpha[-1:]], axis=0)  # (T-1, n_key, 3)

    theta0_key = theta_key[0]                                  # (n_key, 3)
    return theta0_key, omega0_key, alpha_key_init


# ══════════════════════════════════════════════════════════════════════════════
# Section 4: Gaussian pulse model (optional)
# ══════════════════════════════════════════════════════════════════════════════

def gaussian_pulses_fwd(
    t_frames: torch.Tensor,
    amplitudes: torch.Tensor,
    means: torch.Tensor,
    log_widths: torch.Tensor,
) -> torch.Tensor:
    """
    Forward acceleration as a sum of K Gaussian pulses.

    a_fwd(t) = Σ_k  α_k · exp(-(t - μ_k)² / (2 σ_k²))

    Physical interpretation: each Gaussian corresponds to a locomotion
    event — push-off, acceleration burst, deceleration — that produces a
    smooth impulse in forward speed.

    Parameters
    ----------
    t_frames   : (T-1,) frame indices as float
    amplitudes : (K,)   pulse amplitudes α_k  (m/s² scale)
    means      : (K,)   pulse centers μ_k in frame units
    log_widths : (K,)   log of σ_k in frames; σ = exp(log_σ)

    Returns
    -------
    a_fwd : (T-1,) forward acceleration at each control step
    """
    sigma = torch.exp(log_widths).clamp(min=0.5)   # (K,) minimum 0.5 frames
    t  = t_frames.unsqueeze(1)                      # (T-1, 1)
    mu = means.unsqueeze(0)                         # (1, K)
    al = amplitudes.unsqueeze(0)                    # (1, K)
    sg = sigma.unsqueeze(0)                         # (1, K)
    gauss = al * torch.exp(-0.5 * ((t - mu) / sg) ** 2)  # (T-1, K)
    return gauss.sum(dim=1)                         # (T-1,)


def fit_pulse_amplitudes(
    a_ctrl_init_np: np.ndarray,
    sprint_dir_np: np.ndarray,
    K: int,
    mu_np: np.ndarray,
    logw_val: float,
) -> np.ndarray:
    """
    Fit K Gaussian pulse amplitudes to the initial forward acceleration signal
    via least-squares.  Amplitudes are unconstrained (positive = acceleration,
    negative = deceleration) — this is the key difference from initializing all
    amplitudes at zero, which forces the pulses to learn from scratch.

    Parameters
    ----------
    a_ctrl_init_np : (T-1, 3)  initial acceleration controls (from smoothed FD)
    sprint_dir_np  : (3,)       sprint direction unit vector
    K              : int         number of pulses
    mu_np          : (K,)        initial pulse center positions (frame indices)
    logw_val       : float       log(σ) — initial log-width for all pulses

    Returns
    -------
    alpha : (K,) float32   fitted pulse amplitudes (positive or negative)
    """
    T1 = a_ctrl_init_np.shape[0]  # T-1
    d  = sprint_dir_np / (np.linalg.norm(sprint_dir_np) + 1e-8)
    a_fwd = (a_ctrl_init_np @ d).astype(np.float64)  # (T-1,)

    sigma = np.exp(logw_val)
    t_frames = np.arange(T1, dtype=np.float64)
    # Gaussian basis matrix (T-1, K)
    Phi = np.exp(-0.5 * ((t_frames[:, None] - mu_np[None, :]) / sigma) ** 2)

    # Least-squares: Phi @ alpha ≈ a_fwd (T1 >> K → overdetermined)
    alpha, _, _, _ = np.linalg.lstsq(Phi, a_fwd, rcond=None)
    print(f"[INFO] Pulse amp init (K={K}): {alpha.round(2)}  "
          f"(+={( alpha>0).sum()} pulses,  -={(alpha<0).sum()} pulses)")
    return alpha.astype(np.float32)


def build_a_ctrl_from_pulses(
    amplitudes: torch.Tensor,
    means: torch.Tensor,
    log_widths: torch.Tensor,
    a_lat: torch.Tensor,
    sprint_dir_t: torch.Tensor,
    T: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build the full (T-1, 3) acceleration control from Gaussian pulse parameters.

    a_ctrl[t] = a_fwd(t) · d   +   a_lat[t]

    where d is the sprint direction unit vector, a_fwd(t) is the forward
    Gaussian pulse acceleration, and a_lat is a small per-frame lateral term.

    Parameters
    ----------
    amplitudes, means, log_widths : pulse parameters (K,)
    a_lat        : (T-1, 2) lateral acceleration (perpendicular to sprint dir)
    sprint_dir_t : (3,)     sprint direction unit vector
    T            : total number of frames
    device       : torch device

    Returns
    -------
    a_ctrl : (T-1, 3)
    """
    t_frames = torch.arange(T - 1, dtype=torch.float32, device=device)
    a_fwd_scalar = gaussian_pulses_fwd(t_frames, amplitudes, means, log_widths)  # (T-1,)

    # Forward component: broadcast along sprint direction
    d = sprint_dir_t / sprint_dir_t.norm().clamp(min=1e-8)         # (3,)
    a_fwd_3d = a_fwd_scalar.unsqueeze(-1) * d.unsqueeze(0)         # (T-1, 3)

    # Lateral component: a_lat lives in the plane perpendicular to d.
    # We provide it as a 2-vector in a local frame; here we just use the
    # first two world axes projected orthogonally to d.
    # For simplicity, add a_lat directly in 3D world space but regularized
    # to be small — the control prior loss handles orthogonality.
    if a_lat.shape[-1] == 3:
        a_lat_3d = a_lat
    else:
        # a_lat is (T-1, 2): embed in 3D as (x, 0, z) correction, will be
        # regularized by the lateral loss to stay small
        a_lat_3d = torch.cat([a_lat[:, 0:1],
                               torch.zeros(T - 1, 1, device=device),
                               a_lat[:, 1:2]], dim=-1)

    return a_fwd_3d + a_lat_3d                                     # (T-1, 3)


# ══════════════════════════════════════════════════════════════════════════════
# Section 4b: Sprint direction estimation
# ══════════════════════════════════════════════════════════════════════════════

def estimate_sprint_dir_from_sam3d(go_world_np: np.ndarray) -> np.ndarray:
    """
    Estimate the sprint direction from SAM3D world-frame global orientation.

    **Why prefer this over PCA on the trajectory?**

    The height-scale trajectory has large per-frame depth noise (~5–20 cm at
    15–20 m depth), so the XZ positions oscillate substantially.  PCA on these
    noisy positions gives an unreliable sprint direction.

    SAM3D estimates body orientation directly from 2D body keypoints (head,
    shoulders, hips), which are less sensitive to depth noise.  The body's
    *facing direction* in a sprint is the direction it is *moving* — using
    the orientation from SAM3D is therefore a more direct and stable estimate.

    Procedure
    ---------
    1.  Convert SAM3D world-frame axis-angle orientations to rotation matrices.
    2.  The body's canonical forward vector is $\\hat{z} = [0, 0, 1]^\\top$.
        In world frame: $\\mathbf{f}_t = R_t \\hat{z}$ = third column of $R_t$.
    3.  Project each $\\mathbf{f}_t$ onto the horizontal (XZ) plane by zeroing
        the Y component, then normalize.
    4.  Take the circular mean (average of unit vectors, then renormalize).
        This is robust to per-frame orientation noise as long as the athlete
        faces roughly the same direction throughout the sequence.

    Parameters
    ----------
    go_world_np : (T, 3)  SAM3D world-frame global orientation (axis-angle)

    Returns
    -------
    d : (3,)  sprint direction unit vector with Y = 0 (horizontal plane only)
    """
    R = rotvec_to_matrix_np(go_world_np)   # (T, 3, 3)

    # R @ [0, 0, 1] = third column of R
    fwd_world = R[:, :, 2]                 # (T, 3)

    # Project to XZ plane (Y-up world: set Y component to zero)
    fwd_xz = np.stack([
        fwd_world[:, 0],
        np.zeros(len(fwd_world), dtype=np.float32),
        fwd_world[:, 2],
    ], axis=-1).astype(np.float32)         # (T, 3) with Y = 0

    # Normalize each frame's projection; discard near-zero frames
    norms  = np.linalg.norm(fwd_xz, axis=-1)          # (T,)
    valid  = norms > 1e-4
    if not valid.any():
        # Degenerate: all orientations point straight up; return default
        warnings.warn("SAM3D forward directions all near-zero on XZ — defaulting to +X")
        return np.array([1., 0., 0.], dtype=np.float32)

    fwd_normed = fwd_xz[valid] / norms[valid, None]    # (N, 3)

    # Circular mean: sum unit vectors then renormalize
    mean_dir  = fwd_normed.mean(axis=0)                # (3,)
    mean_norm = np.linalg.norm(mean_dir)
    if mean_norm < 1e-6:
        # Circular mean collapsed (bi-directional sprinting?): use first valid
        mean_dir  = fwd_normed[0]
        mean_norm = np.linalg.norm(mean_dir) + 1e-8

    d = (mean_dir / mean_norm).astype(np.float32)
    print(f"[INFO] Sprint direction from SAM3D orient: d={d.round(4)}")
    return d


# ══════════════════════════════════════════════════════════════════════════════
# Section 5: Loss functions (rollout-specific)
# ══════════════════════════════════════════════════════════════════════════════

def loss_control_lateral(
    a_ctrl: torch.Tensor,
    sprint_dir_t: torch.Tensor,
    normalized: bool = True,
) -> torch.Tensor:
    """
    Lateral (cross-track) acceleration penalty.

    Normalized form (default, recommended):
        L_lat = mean_t  ||a_perp||² / ||a||²

    This is the fraction of kinetic control energy that is lateral, so
    the loss lives in [0, 1] regardless of the magnitude of a_ctrl.  This
    avoids the weight-scale sensitivity problem: when a_ctrl is initialized
    from noisy finite differences (magnitudes ~100-1000 m/s²), the raw squared
    form ||a_perp||² can be O(10⁵), completely drowning the reprojection loss.
    The normalized form is always O(1), making w_lat interpretable as a
    soft constraint: w_lat=1 says "lateral energy should be ≤ reproj energy."

    Raw form (normalized=False):
        L_lat = mean_t  ||a_perp||²

    Parameters
    ----------
    a_ctrl      : (T-1, 3)
    sprint_dir_t: (3,)
    normalized  : bool  if True (default) use scale-invariant ratio form

    Returns
    -------
    scalar loss in [0, 1] (normalized) or [0, ∞) (raw)
    """
    d = sprint_dir_t / sprint_dir_t.norm().clamp(min=1e-8)
    a_par  = (a_ctrl @ d).unsqueeze(-1) * d.unsqueeze(0)   # (T-1, 3) parallel
    a_perp = a_ctrl - a_par                                 # (T-1, 3) lateral
    lat_sq = (a_perp ** 2).sum(dim=-1)                     # (T-1,)

    if normalized:
        total_sq = (a_ctrl ** 2).sum(dim=-1).clamp(min=1e-4)  # (T-1,)
        return (lat_sq / total_sq).mean()
    else:
        return lat_sq.mean()


def loss_control_smooth_fwd(
    a_ctrl: torch.Tensor,
    sprint_dir_t: torch.Tensor,
) -> torch.Tensor:
    """
    Forward acceleration smoothness (first-order temporal regularizer on controls).

    L_smooth = mean_t  (a_{t+1}·d - a_t·d)²

    In a real sprint, the forward acceleration changes smoothly between
    push-off and air-borne phases — no instantaneous spikes.

    Parameters
    ----------
    a_ctrl      : (T-1, 3)
    sprint_dir_t: (3,)

    Returns
    -------
    scalar loss
    """
    d = sprint_dir_t / sprint_dir_t.norm().clamp(min=1e-8)
    a_fwd = a_ctrl @ d                                     # (T-1,) forward component
    if a_fwd.shape[0] < 2:
        return a_ctrl.new_zeros(1).squeeze()
    da_fwd = a_fwd[1:] - a_fwd[:-1]                       # (T-2,)
    return (da_fwd ** 2).mean()


def loss_ctrl_jerk(a_ctrl: torch.Tensor) -> torch.Tensor:
    """
    Full 3-D control jerk penalty — penalizes the rate of change of the
    acceleration control vector in all three directions.

    L_ctrl_jerk = mean_t  ||a_{t+1} - a_t||²

    Why this helps when w_smooth_fwd alone does not
    -----------------------------------------------
    w_smooth_fwd penalizes only the FORWARD component of acceleration change.
    The lateral component is free to oscillate every frame in response to
    per-frame 2D observation noise, producing high jerk even when the forward
    acceleration is perfectly smooth.  This term closes that gap.

    Parameters
    ----------
    a_ctrl : (T-1, 3)

    Returns
    -------
    scalar loss
    """
    if a_ctrl.shape[0] < 2:
        return a_ctrl.new_zeros(1).squeeze()
    da = a_ctrl[1:] - a_ctrl[:-1]          # (T-2, 3)
    return (da ** 2).sum(dim=-1).mean()


def loss_lateral_velocity(
    v_seq: torch.Tensor,
    sprint_dir_t: torch.Tensor,
    normalized: bool = True,
) -> torch.Tensor:
    """
    Lateral velocity penalty — directly suppresses cross-track motion.

    Normalized form (default):
        L_lat_vel = mean_t  ||v_t^perp||² / ||v_t||²

    Why this is needed alongside loss_control_lateral
    -------------------------------------------------
    loss_control_lateral penalizes lateral *acceleration*, which only prevents
    future lateral velocity from growing.  It does NOT fix the lateral component
    of the initial velocity v0 (which is set from the noisy height-scale finite
    difference and may already be 1–2 m/s lateral).

    Because velocity integrates: v_{t+1} = v_t + a_t * dt, a large lateral v0
    propagates through the entire trajectory unchanged even if all future a_t
    are perfectly forward.  This loss directly penalizes the lateral velocity at
    every frame, forcing the optimizer to also correct v0 and any velocity drift.

    Effect on metrics
    -----------------
    • sprint_align (cos(v, d̂)) → increases toward 1.0
    • v_lat_mean → decreases toward 0
    • The reproj loss remains satisfied because 2D reprojection is insensitive
      to overall depth scaling — the optimizer can reduce depth while keeping
      the 2D projection constant.

    Parameters
    ----------
    v_seq        : (T, 3)
    sprint_dir_t : (3,)
    normalized   : bool  if True, ratio form ∈ [0, 1]

    Returns
    -------
    scalar loss
    """
    d = sprint_dir_t / sprint_dir_t.norm().clamp(min=1e-8)
    v_par  = (v_seq @ d).unsqueeze(-1) * d.unsqueeze(0)   # (T, 3) parallel component
    v_perp = v_seq - v_par                                  # (T, 3) lateral component
    lat_sq = (v_perp ** 2).sum(dim=-1)                     # (T,)

    if normalized:
        speed_sq = (v_seq ** 2).sum(dim=-1).clamp(min=1e-4)
        return (lat_sq / speed_sq).mean()
    else:
        return lat_sq.mean()


def loss_control_yaw(yaw_acc_ctrl: torch.Tensor) -> torch.Tensor:
    """
    Yaw angular acceleration regularizer.

    L_yaw = mean_t  α_t²

    In a straight sprint the heading barely changes, so α_t should be near zero.
    Regularizing it prevents spurious turning artifacts from noisy SAM3D yaw.

    Parameters
    ----------
    yaw_acc_ctrl : (T-1,)

    Returns
    -------
    scalar loss
    """
    return (yaw_acc_ctrl ** 2).mean()


def loss_anchor_translation(
    x_rollout: torch.Tensor,
    x_anchor: torch.Tensor,
    frame_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Weak L2 anchor to the SAM3D / height-scale translation.

    L_anchor = mean_t  w_t * ||x_t - x_t^anchor||²

    Prevents the rollout trajectory from drifting arbitrarily far from the
    initial estimate.  Use a small weight (0.01–0.1) so observations dominate.

    Parameters
    ----------
    x_rollout     : (T, 3)  rolled-out translation sequence
    x_anchor      : (T, 3)  initial (height-scale / SAM3D) translation  (fixed)
    frame_weights : (T,) optional per-frame confidence weights

    Returns
    -------
    scalar loss
    """
    sq = ((x_rollout - x_anchor) ** 2).sum(dim=-1)  # (T,)
    if frame_weights is not None:
        sq = sq * frame_weights
    return sq.mean()


def loss_anchor_yaw(
    yaw_rollout: torch.Tensor,
    yaw_anchor: torch.Tensor,
    frame_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Weak L2 anchor to the SAM3D yaw sequence.

    L_anchor_yaw = mean_t  w_t * (ψ_t - ψ_t^anchor)²

    Parameters
    ----------
    yaw_rollout   : (T,)
    yaw_anchor    : (T,)  fixed SAM3D yaw estimates
    frame_weights : (T,) optional per-frame confidence weights

    Returns
    -------
    scalar loss
    """
    sq = (yaw_rollout - yaw_anchor) ** 2  # (T,)
    if frame_weights is not None:
        sq = sq * frame_weights
    return sq.mean()


def loss_orient_gravity_align(
    R_seq: torch.Tensor,
    world_up: torch.Tensor,
) -> torch.Tensor:
    """
    Gravity alignment: penalize the spine pointing away from vertical.

    spine_up = R @ [0,1,0] = second column of R.
    world_up is computed once from mean initial SAM3D spine directions.

    Loss = 1 - cos(spine_up, world_up)
           = 1 - (spine_up · world_up)

    Value: 0 = perfectly upright, 2 = fully inverted.
    A sprinting athlete should have ~0.03-0.15 (slight forward lean).
    """
    spine_up = R_seq[:, :, 1]  # (T, 3): second column = R @ e_y
    cos_align = (spine_up * world_up.unsqueeze(0)).sum(dim=-1)  # (T,)
    return (1.0 - cos_align).mean()


def loss_orient_smooth(alpha3d: torch.Tensor) -> torch.Tensor:
    """
    3D angular acceleration regularizer (full_dof orientation).
    Analogous to loss_control_yaw for yaw-only mode.

    L = mean ||alpha3d_t||^2
    """
    return (alpha3d ** 2).sum(dim=-1).mean()


def loss_orient_yaw_align(
    R_seq: torch.Tensor,
    sprint_dir: torch.Tensor,
    world_up: torch.Tensor,
) -> torch.Tensor:
    """
    Soft yaw alignment: penalises chest_fwd deviating from sprint_dir
    in the horizontal plane.

    Separates yaw constraint from pitch/roll — complements loss_orient_gravity_align
    which constrains pitch/roll (spine stays vertical).

    L = mean(1 - cos(chest_fwd_horizontal, sprint_dir))  in [0, 2]

    R_seq    : (T, 3, 3)
    sprint_dir: (3,)  horizontal unit vector
    world_up  : (3,)  vertical unit vector
    """
    chest_fwd = R_seq[:, :, 2]                                          # (T, 3)
    # Project chest_fwd to horizontal plane (remove world_up component)
    vert_comp  = (chest_fwd * world_up.unsqueeze(0)).sum(dim=-1, keepdim=True)  # (T,1)
    chest_h    = chest_fwd - vert_comp * world_up.unsqueeze(0)          # (T, 3)
    chest_h    = chest_h / chest_h.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cos_align  = (chest_h * sprint_dir.unsqueeze(0)).sum(dim=-1)        # (T,)
    return (1.0 - cos_align).mean()


# ── Biomechanics losses ───────────────────────────────────────────────────────

def compute_contact_labels_np(
    J_world_np: np.ndarray,
    floor_h: float,
    foot_ids: List[int] = None,
    margin: float = 0.08,
    kernel_size: int = 5,
    world_up_vec: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Detect stance vs flight frames from foot geometry.

    Uses projection onto world_up for height (works with any Y-axis orientation).
    If world_up_vec is None, falls back to raw Y component.

    Returns φ ∈ {0,1} shape (T,): 1=stance (foot within margin of floor), 0=flight.
    Median-filtered to remove single-frame flickers.
    """
    if foot_ids is None:
        foot_ids = _FOOT_JOINT_IDS
    if world_up_vec is not None:
        wu = world_up_vec / (np.linalg.norm(world_up_vec) + 1e-8)
        foot_pts = J_world_np[:, foot_ids, :]                  # (T, 4, 3)
        foot_h   = (foot_pts * wu[None, None, :]).sum(-1).min(axis=1)  # (T,) proj onto up
    else:
        foot_h = J_world_np[:, foot_ids, 1].min(axis=1)       # (T,) raw Y fallback
    phi = (foot_h <= floor_h + margin).astype(np.float32)
    if _SCIPY_SIGNAL_OK and len(phi) >= kernel_size:
        phi = _medfilt(phi, kernel_size=kernel_size)
    return phi


def loss_wbam(
    J_world: torch.Tensor,
    dt: float,
    phi_t: Optional[torch.Tensor] = None,
    masses: Optional[torch.Tensor] = None,
    w_stance: float = 1.0,
    w_flight: float = 1.0,
) -> torch.Tensor:
    """
    Whole-body angular momentum (WBAM) near zero.

    Enforces the biomechanical regulation law (Herr & Popov 2008, Hinrichs 1987):
    arm–leg opposition keeps WBAM near zero throughout the gait cycle.

    Gradient flows through J_world → SMPL(global_orient, body_pose_opt, transl),
    coupling translation, global orientation, and local joint rotations.

    J_world : (T, 24, 3) world-frame SMPL joints
    dt      : time step
    phi_t   : (T,) stance labels [0,1]; if None, uniform weighting
    masses  : (24,) segment mass fractions; if None, use Winter-1990 approx
    """
    if masses is None:
        masses = torch.tensor(_SMPL24_SEG_MASSES_NP, device=J_world.device,
                              dtype=J_world.dtype)

    # Mass-weighted CoM
    r_CoM = (J_world * masses[None, :, None]).sum(dim=1) / masses.sum()  # (T, 3)

    # Finite-diff joint velocities
    v_J   = (J_world[1:] - J_world[:-1]) / dt               # (T-1, 24, 3)
    r_mid = (J_world[:-1] + J_world[1:]) / 2.0              # (T-1, 24, 3) midpoint
    r_rel = r_mid - r_CoM[:-1].unsqueeze(1)                  # (T-1, 24, 3) relative to CoM

    # WBAM per frame: Σ_i m_i * (r_i × v_i)
    L_per  = torch.cross(r_rel, v_J, dim=-1)                 # (T-1, 24, 3)
    L_wbam = (L_per * masses[None, :, None]).sum(dim=1)      # (T-1, 3)
    L_mag  = L_wbam.norm(dim=-1)                             # (T-1,)

    if phi_t is not None and phi_t.shape[0] > 1:
        phi_mid   = (phi_t[:-1] + phi_t[1:]) / 2.0          # (T-1,)
        frame_w   = phi_mid * w_stance + (1.0 - phi_mid) * w_flight
        return (frame_w * L_mag).mean()

    return L_mag.mean()


def loss_orient_contact(
    alpha3d: torch.Tensor,
    phi_t: torch.Tensor,
    w_stance: float = 1.0,
    w_flight: float = 0.1,
) -> torch.Tensor:
    """
    Contact-conditioned global orientation smoothness.

    Replaces the uniform loss_orient_smooth: stance frames (foot planted) receive
    tighter regularization on angular acceleration; flight frames are looser
    (body can rotate more freely, as no external torque is applied).

    alpha3d : (T-1, 3) global angular acceleration controls
    phi_t   : (T-1,)   stance labels aligned to alpha3d frames
    """
    ang_acc_sq = (alpha3d ** 2).sum(dim=-1)                  # (T-1,)
    frame_w = phi_t * w_stance + (1.0 - phi_t) * w_flight
    return (frame_w * ang_acc_sq).mean()


def loss_local_contact(
    alpha_key: torch.Tensor,
    phi_t: torch.Tensor,
    w_stance: float = 1.0,
    w_flight: float = 0.1,
) -> torch.Tensor:
    """
    Contact-conditioned local joint angular acceleration smoothness.

    During stance the limbs are constrained (foot is planted, kinematic chain
    is anchored); during flight the swing phase allows larger accelerations.

    alpha_key : (T-1, n_key, 3)
    phi_t     : (T-1,) stance labels
    """
    ang_acc_sq = (alpha_key ** 2).sum(dim=-1).sum(dim=-1)    # (T-1,)
    frame_w    = phi_t * w_stance + (1.0 - phi_t) * w_flight
    return (frame_w * ang_acc_sq).mean()


def loss_local_anchor(
    theta_key_seq: torch.Tensor,
    body_pose_key_anchor: torch.Tensor,
) -> torch.Tensor:
    """
    L2 pull of optimized local joint angles toward SAM3D observations.
    Prevents local rotations from drifting into implausible pose space.

    theta_key_seq      : (T, n_key, 3) optimized
    body_pose_key_anchor: (T, n_key, 3) from SAM3D (fixed)
    """
    return ((theta_key_seq - body_pose_key_anchor) ** 2).mean()


def build_body_pose_opt(
    body_pose_base: torch.Tensor,
    theta_key_seq: torch.Tensor,
) -> torch.Tensor:
    """
    Construct the full (T, 69) body_pose tensor by replacing key joint angles
    with the optimized values, keeping non-key joints fixed from SAM3D.

    Uses sorted chunk concatenation so gradients flow through theta_key_seq
    while non-key chunks have no gradient (body_pose_base is a fixed tensor).

    body_pose_base : (T, 69)  fixed SAM3D body_pose
    theta_key_seq  : (T, n_key, 3)  optimized key joint axis-angles
    """
    pieces = []
    cursor = 0
    for i, bp_s in enumerate(_KEY_BP_STARTS):
        if bp_s > cursor:
            pieces.append(body_pose_base[:, cursor:bp_s])
        pieces.append(theta_key_seq[:, i, :])
        cursor = bp_s + 3
    if cursor < 69:
        pieces.append(body_pose_base[:, cursor:])
    return torch.cat(pieces, dim=1)                          # (T, 69)


# ══════════════════════════════════════════════════════════════════════════════
# Section 6: Initialization from height-scale / SAM3D
# ══════════════════════════════════════════════════════════════════════════════

def init_rollout_from_world_traj(
    transl_world_init: np.ndarray,
    go_world_init: np.ndarray,
    dt: float,
    smooth_sigma: float = 3.0,
) -> Dict[str, np.ndarray]:
    """
    Derive rollout initial state and initial controls from the height-scale
    trajectory (the cleanest available prior).

    Procedure
    ---------
    1. Smooth the raw trajectory with a Gaussian filter (sigma frames) to remove
       per-frame depth-estimation jitter before computing derivatives.
       Without smoothing, the height-scale trajectory has ~5-20 cm per-frame
       noise at 15-20 m depth, which propagates through finite differences as
       ~100-1000 m/s² acceleration noise — far larger than the true sprint
       acceleration of ~1-5 m/s².
    2. x0 = transl_world_init[0]  (from UNSMOOTHED, i.e. true initial position)
    3. v0 ≈ finite difference of SMOOTHED transl at t=0
    4. a_ctrl_init[t] = second finite difference of SMOOTHED transl
    5. yaw0, yaw_rate0, yaw_acc_init extracted from go_world_init rotations

    Parameters
    ----------
    transl_world_init : (T, 3)  height-scale world translation
    go_world_init     : (T, 3)  SAM3D world global orientation (axis-angle)
    dt                : float   time step
    smooth_sigma      : float   Gaussian sigma (frames) for trajectory pre-smoothing.
                                0 disables smoothing (uses raw finite differences).
                                3.0 (default) ≈ 0.1 s at 30 fps — removes jitter
                                without blurring real motion events.

    Returns
    -------
    dict with keys:
        x0, v0              : (3,) initial state
        yaw0, yaw_rate0     : scalars
        a_ctrl_init         : (T-1, 3)  initial acceleration controls
        yaw_acc_ctrl_init   : (T-1,)    initial yaw controls
        yaw_seq_init        : (T,)      yaw angles from SAM3D
        a_ctrl_mag_mean     : float     mean magnitude of a_ctrl_init (for diagnostics)
    """
    try:
        from scipy.ndimage import gaussian_filter1d as _gf1d
        _have_scipy = True
    except ImportError:
        _have_scipy = False
        warnings.warn("scipy not found — skipping trajectory smoothing in init. "
                      "Expect large a_ctrl_init magnitudes.", stacklevel=2)

    T = transl_world_init.shape[0]

    # ── Smooth trajectory to suppress per-frame depth jitter ──────────────
    if smooth_sigma > 0 and _have_scipy:
        transl_smooth = _gf1d(
            transl_world_init.astype(np.float64),
            sigma=smooth_sigma, axis=0,
        ).astype(np.float32)
        print(f"[INFO] Init smoothing: sigma={smooth_sigma:.1f} frames  "
              f"raw_a_mag={np.linalg.norm((transl_world_init[2:] - 2*transl_world_init[1:-1] + transl_world_init[:-2]) / dt**2, axis=-1).mean():.1f} m/s²  ", end="")
    else:
        transl_smooth = transl_world_init.copy()

    # ── Translation ───────────────────────────────────────────────────────
    # Use the raw first position (not smoothed) so the rollout starts from
    # the correct observed position.
    x0 = transl_world_init[0].copy()

    # Velocity from first-order finite difference of smoothed trajectory
    vel = (transl_smooth[1:] - transl_smooth[:-1]) / dt             # (T-1, 3)
    v0  = vel[0].copy()

    # Acceleration from second-order finite difference of smoothed trajectory
    acc = (vel[1:] - vel[:-1]) / dt                                  # (T-2, 3)
    # a_ctrl_init: T-1 values — pad the last one by repeating
    a_ctrl_init = np.concatenate([acc, acc[-1:]], axis=0)            # (T-1, 3)

    a_mag_mean = float(np.linalg.norm(a_ctrl_init, axis=-1).mean())
    if smooth_sigma > 0 and _have_scipy:
        print(f"smoothed_a_mag={a_mag_mean:.2f} m/s²")
    else:
        print(f"[INFO] a_ctrl_init mean magnitude: {a_mag_mean:.2f} m/s²")

    # ── Yaw from SAM3D root orientation ───────────────────────────────────
    R_world = rotvec_to_matrix_np(go_world_init)                     # (T, 3, 3)
    fwd_x   = R_world[:, 0, 2]                                       # forward proj X
    fwd_z   = R_world[:, 2, 2]                                       # forward proj Z
    yaw_seq = np.arctan2(fwd_x, fwd_z).astype(np.float32)           # (T,)

    yaw0      = float(yaw_seq[0])
    yaw_rates = (yaw_seq[1:] - yaw_seq[:-1]) / dt                   # (T-1,)
    yaw_rate0 = float(yaw_rates[0])

    yaw_acc   = (yaw_rates[1:] - yaw_rates[:-1]) / dt               # (T-2,)
    yaw_acc_ctrl_init = np.append(yaw_acc, yaw_acc[-1])              # (T-1,)

    return {
        "x0":               x0.astype(np.float32),
        "v0":               v0.astype(np.float32),
        "yaw0":             np.float32(yaw0),
        "yaw_rate0":        np.float32(yaw_rate0),
        "a_ctrl_init":      a_ctrl_init.astype(np.float32),
        "yaw_acc_ctrl_init":yaw_acc_ctrl_init.astype(np.float32),
        "yaw_seq_init":     yaw_seq,
        "a_ctrl_mag_mean":  a_mag_mean,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Section 7: Main optimization loop
# ══════════════════════════════════════════════════════════════════════════════

def run_rollout_optimization(
    smpl_model,
    body_pose_np: np.ndarray,
    betas_np: np.ndarray,
    obs2d_np: np.ndarray,
    focal_np: np.ndarray,
    cx: float,
    cy: float,
    dt: float,
    transl_world_init: np.ndarray,
    go_world_init: np.ndarray,
    cam_rvec_init: np.ndarray,
    cam_tvec_init: np.ndarray,
    sprint_dir_np: Optional[np.ndarray],
    args,
    device: torch.device,
    obs2d_conf_np: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Core Adam optimization loop for rollout-based human motion reconstruction.

    Variables optimized
    -------------------
    Phase 1 (iters 0 .. stage1_iters-1): only initial state + camera
        x0, v0, yaw0, yaw_rate0, [cam_rvec, cam_tvec]
    Phase 2 (stage1_iters .. iters-1): full joint
        + a_ctrl, yaw_acc_ctrl

    Phase-1 focus: align initial state with observations.
    Phase-2 focus: fine-tune the full control trajectory.

    Returns
    -------
    results dict stored under data["joint_cam_human_optim"] for eval script compatibility
    """
    T  = body_pose_np.shape[0]

    # ── Fixed tensors ──────────────────────────────────────────────────────
    body_pose_t = to_torch(body_pose_np, device)       # (T, 69) — fixed SAM3D base
    betas_t     = to_torch(betas_np, device)           # (T, B)

    # ── Local rotation mode setup ─────────────────────────────────────────
    use_local_rot = (getattr(args, 'local_rot_mode', 'fixed') == 'velocity_rollout')
    if use_local_rot:
        _body_pose_key_np = np.stack(
            [body_pose_np[:, bp_s:bp_s+3] for bp_s in _KEY_BP_STARTS], axis=1
        ).astype(np.float32)                               # (T, n_key, 3)
        body_pose_key_anchor_t = to_torch(_body_pose_key_np, device)
    else:
        body_pose_key_anchor_t = None
    obs2d_t     = to_torch(obs2d_np, device)           # (T, 24, 2)
    focal_t     = to_torch(focal_np, device)           # (T,)

    # Per-frame confidence weights from pose fit quality (None = uniform)
    obs2d_conf_t: Optional[torch.Tensor] = None
    if obs2d_conf_np is not None:
        obs2d_conf_t = to_torch(obs2d_conf_np, device)  # (T,)

    # SAM3D root rotation matrices (for pitch/roll extraction, fixed)
    R_sam_world = to_torch(
        rotvec_to_matrix_np(go_world_init), device
    )                                                  # (T, 3, 3)

    # World-up direction from initial SAM3D orientations (spine_up = R@[0,1,0])
    spine_up_init = R_sam_world[:, :, 1]  # (T, 3)
    world_up = spine_up_init.mean(dim=0)
    world_up = world_up / world_up.norm().clamp(min=1e-8)
    # Log for diagnostics
    print(f"[INFO] World-up from SAM3D init: {world_up.cpu().numpy().round(3)}")

    # SAM3D yaw sequence for anchor
    yaw_sam_t = to_torch(
        extract_yaw_from_rotmat(R_sam_world).cpu().numpy()
        if not isinstance(R_sam_world, np.ndarray)
        else np.arctan2(R_sam_world[:, 0, 2], R_sam_world[:, 2, 2]),
        device,
    )                                                  # (T,)

    # Anchor translation (height-scale init)
    x_anchor_t = to_torch(transl_world_init, device)  # (T, 3)

    # Sprint direction (fixed unit vector)
    if sprint_dir_np is not None:
        sprint_dir_t = to_torch(sprint_dir_np, device)   # (3,)
        has_sprint = True
    else:
        sprint_dir_t = torch.tensor([0., 0., 1.], device=device)
        has_sprint = False

    # ── Derive rollout initialization ─────────────────────────────────────
    init = init_rollout_from_world_traj(
        transl_world_init, go_world_init, dt,
        smooth_sigma=float(getattr(args, 'smooth_a_ctrl_sigma', 3.0)),
    )

    # ── iters=0: return init trajectory without any optimization ──────────
    # Useful as the "SAM3D init only" baseline in ablations — shows exactly
    # what the SAM3D estimate looks like before any rollout optimization.
    if args.iters == 0:
        print("[INFO] iters=0 — returning SAM3D init trajectory (no optimization).")
        with torch.no_grad():
            x0_t   = to_torch(init["x0"],   device)
            v0_t   = to_torch(init["v0"],   device)
            yaw0_t = to_torch(np.array([init["yaw0"]]),      device)
            yr0_t  = to_torch(np.array([init["yaw_rate0"]]), device)
            ac_t   = to_torch(init["a_ctrl_init"],           device)
            ya_t   = to_torch(init["yaw_acc_ctrl_init"],     device)
            x_init_f, v_init_f, yaw_init_f, _ = rollout_states(
                x0_t, v0_t, yaw0_t, yr0_t, ac_t, ya_t, dt)
            R_init_f   = combine_yaw_with_sam3d(yaw_init_f, R_sam_world)
            go_init_f  = rotmat_to_rotvec_diff(R_init_f)
            J_init_f   = run_smpl(smpl_model, go_init_f, body_pose_t,
                                  betas_t, x_init_f, device)
            R_cam_f    = torch.eye(3, device=device).unsqueeze(0).expand(T, 3, 3)
            cam_tvec_f = torch.zeros(T, 3, device=device)
            uv_f       = project_joints(J_init_f, R_cam_f, cam_tvec_f, focal_t, cx, cy)
            reproj_err = (uv_f - obs2d_t).norm(dim=-1).cpu().numpy()
        return {
            "transl_world":    x_init_f.cpu().numpy(),
            "go_world":        go_init_f.cpu().numpy(),
            "v_seq":           v_init_f.cpu().numpy(),
            "yaw_seq":         yaw_init_f.cpu().numpy(),
            "a_ctrl":          init["a_ctrl_init"],
            "cam_rvec":        np.zeros((T, 3), dtype=np.float32),
            "cam_tvec":        np.zeros((T, 3), dtype=np.float32),
            "cam_R":           np.tile(np.eye(3, dtype=np.float32)[None], (T, 1, 1)),
            "focal":           focal_np.copy(),
            "floor_y":         0.0,
            "sprint_dir":      sprint_dir_np,
            "pulse_info":      None,
            "loss_logs":       {},
            "reproj_err_mean": float(reproj_err.mean()),
            "reproj_err_p90":  float(np.percentile(reproj_err, 90)),
        }

    # ── Optimization variables ────────────────────────────────────────────
    x0_var         = torch.nn.Parameter(to_torch(init["x0"],              device))
    v0_var         = torch.nn.Parameter(to_torch(init["v0"],              device))
    yaw0_var       = torch.nn.Parameter(to_torch(np.array([init["yaw0"]]),device))
    yaw_rate0_var  = torch.nn.Parameter(to_torch(np.array([init["yaw_rate0"]]), device))

    # For phase-1 we keep a_ctrl frozen; in phase-2 we unfreeze it
    a_ctrl_var = torch.nn.Parameter(to_torch(init["a_ctrl_init"], device))       # (T-1, 3)
    yaw_acc_var = torch.nn.Parameter(to_torch(init["yaw_acc_ctrl_init"], device)) # (T-1,)

    # ── Full-DOF orientation variables (only when orient_mode == 'full_dof') ──
    use_full_dof_orient = (getattr(args, 'orient_mode', 'yaw_sam_pitchroll') == 'full_dof')
    if use_full_dof_orient:
        _smooth_s = float(getattr(args, 'smooth_a_ctrl_sigma', 3.0))
        _w3d_0_init, _alpha3d_init = init_full_dof_orient_np(
            go_world_init, dt, smooth_sigma=_smooth_s)
        w3d_0_var    = torch.nn.Parameter(to_torch(_w3d_0_init,  device))   # (3,)
        alpha3d_var  = torch.nn.Parameter(to_torch(_alpha3d_init, device))  # (T-1, 3)
        # Initial orientation from SAM3D frame 0
        R0_full_dof  = R_sam_world[0].clone()                               # (3, 3)
        print(f"[INFO] full_dof orient: w3d_0={_w3d_0_init.round(3)}  "
              f"alpha3d_mag={float(np.linalg.norm(_alpha3d_init, axis=-1).mean()):.3f} rad/s²")
    else:
        w3d_0_var = alpha3d_var = R0_full_dof = None

    # ── Local rotation variables (optional) ───────────────────────────────
    theta0_key_var = omega0_key_var = alpha_key_var = None
    if use_local_rot:
        _smooth_lr = float(getattr(args, 'smooth_a_ctrl_sigma', 3.0))
        _th0, _om0, _al0 = init_local_rot_np(body_pose_np, dt, smooth_sigma=_smooth_lr)
        theta0_key_var = torch.nn.Parameter(to_torch(_th0, device))   # (n_key, 3)
        omega0_key_var = torch.nn.Parameter(to_torch(_om0, device))   # (n_key, 3)
        alpha_key_var  = torch.nn.Parameter(to_torch(_al0, device))   # (T-1, n_key, 3)
        print(f"[INFO] Local rot mode: velocity_rollout  "
              f"key_joints={_KEY_JOINT_SMPL}  n={N_KEY_JOINTS}")

    # ── Contact label state (updated periodically during optimization) ────
    phi_np: Optional[np.ndarray] = None
    phi_t:  Optional[torch.Tensor] = None

    # ── Gaussian pulse model (optional) ──────────────────────────────────
    use_pulse = bool(args.use_pulse_model)
    K = int(args.n_pulses)
    if use_pulse:
        # Pulse centers and widths (same as before)
        pulse_mean = torch.nn.Parameter(
            torch.linspace(T * 0.1, T * 0.9, K, device=device)
        )
        _logw_val = float(np.log(max(T / (K * 3), 1.5)))
        pulse_logw = torch.nn.Parameter(
            torch.full((K,), _logw_val, device=device)
        )
        # Initialize amplitudes via least-squares fit to smoothed init forward
        # acceleration — allows positive (accel) and negative (decel) from the start.
        _mu_np = np.linspace(T * 0.1, T * 0.9, K)
        _amp_init = fit_pulse_amplitudes(
            init["a_ctrl_init"], sprint_dir_np, K, _mu_np, _logw_val,
        )
        pulse_amp  = torch.nn.Parameter(torch.tensor(_amp_init, device=device))
        # Lateral acceleration (small, per-frame, 2D → 3D via embedding)
        a_lat_var  = torch.nn.Parameter(torch.zeros(T - 1, 2, device=device))
        print(f"[INFO] Gaussian pulse model: K={K} pulses"
              f"  init means={pulse_mean.detach().cpu().numpy().round(1)}")
    else:
        pulse_amp = pulse_mean = pulse_logw = a_lat_var = None

    # ── Camera ──────────────────────────────────────────────────────────
    if args.opt_cam:
        cam_rvec_var = torch.nn.Parameter(
            to_torch(cam_rvec_init.mean(axis=0, keepdims=True), device))  # (1, 3)
        cam_tvec_var = torch.nn.Parameter(
            to_torch(cam_tvec_init.mean(axis=0, keepdims=True), device))  # (1, 3)
    else:
        cam_rvec_var = to_torch(np.zeros((1, 3), dtype=np.float32), device)
        cam_tvec_var = to_torch(np.zeros((1, 3), dtype=np.float32), device)

    # ── Floor parameter ────────────────────────────────────────────────
    if args.use_contact_loss:
        floor_y_init = float(np.percentile(transl_world_init[:, 1], 5)
                             - args.target_height_m * 0.55)
        floor_y_var = torch.nn.Parameter(
            torch.tensor([floor_y_init], device=device)
        )
        print(f"[INFO] Contact loss ON  floor_y_init={floor_y_init:.3f}")
    else:
        floor_y_var = torch.tensor([0.0], device=device)

    # ── Staged optimizer setup ─────────────────────────────────────────
    staged      = args.iters > args.stage1_iters and args.stage1_iters > 0
    s1_iters    = args.stage1_iters if staged else 0

    def _make_optimizer(phase2: bool) -> torch.optim.Adam:
        # Phase-1 state params; in full_dof mode swap yaw0/yaw_rate0 for w3d_0
        if use_full_dof_orient:
            p1_params = [x0_var, v0_var, w3d_0_var]
        else:
            p1_params = [x0_var, v0_var, yaw0_var, yaw_rate0_var]
        if use_local_rot:
            p1_params = p1_params + [theta0_key_var, omega0_key_var]
        groups = [{"params": p1_params, "lr": args.lr}]
        if phase2:
            if use_pulse:
                groups.append({"params": [pulse_amp, pulse_mean, pulse_logw], "lr": args.lr})
                groups.append({"params": [a_lat_var], "lr": args.lr * 0.5})
            else:
                groups.append({"params": [a_ctrl_var, yaw_acc_var], "lr": args.lr})
            if use_full_dof_orient:
                groups.append({"params": [alpha3d_var], "lr": args.lr * 0.5})
            if use_local_rot:
                # Local joints much slower — don't stray far from SAM3D observations
                lr_local = args.lr * float(getattr(args, 'local_rot_lr_scale', 0.3))
                groups.append({"params": [alpha_key_var], "lr": lr_local})
        if args.opt_cam:
            groups.append({"params": [cam_rvec_var, cam_tvec_var], "lr": args.lr_cam})
        if args.use_contact_loss:
            groups.append({"params": [floor_y_var], "lr": args.lr * 0.02})
        return torch.optim.Adam(groups)

    optimizer = _make_optimizer(phase2=not staged)

    use_lr_decay = (args.lr_end > 0) and (args.iters > 1)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.iters, eta_min=args.lr_end)
        if use_lr_decay else None)

    # ── Loss weights ──────────────────────────────────────────────────
    w_reproj      = args.w_reproj
    w_contact     = args.w_contact     if args.use_contact_loss else 0.0
    w_lat         = args.w_lat         if has_sprint else 0.0
    w_smooth_fwd  = args.w_smooth_fwd
    w_yaw_ctrl    = args.w_yaw_ctrl
    w_anchor      = args.w_anchor
    w_anchor_yaw  = args.w_anchor_yaw
    w_acc_reg      = getattr(args, 'w_acc_reg', 0.0)
    w_ctrl_jerk    = getattr(args, 'w_ctrl_jerk', 0.0)
    w_lat_vel      = getattr(args, 'w_lat_vel', 0.0)
    w_orient_gravity = getattr(args, 'w_orient_gravity', 0.0)
    w_orient_smooth  = getattr(args, 'w_orient_smooth', 0.0)
    normalize_lat = bool(getattr(args, 'normalize_lat_loss', True))

    # ── Biomechanics loss weights ─────────────────────────────────────────
    w_wbam           = getattr(args, 'w_wbam', 0.0)
    w_orient_contact = getattr(args, 'w_orient_contact', 0.0)
    w_local_contact  = getattr(args, 'w_local_contact', 0.0)
    w_local_anchor   = getattr(args, 'w_local_anchor', 2.0)  if use_local_rot else 0.0
    w_local_jerk     = getattr(args, 'w_local_jerk', 0.05)   if use_local_rot else 0.0
    biom_margin      = float(getattr(args, 'biom_margin', 0.08))
    biom_update_iters= int(getattr(args, 'biom_update_iters', 50))
    # w_orient_contact adds contact-conditioned extra regularization (does not replace smooth)

    print(
        f"[INFO] Weights — reproj={w_reproj}  contact={w_contact}  "
        f"lat={w_lat}{'(norm)' if normalize_lat else '(raw)'}  "
        f"smooth_fwd={w_smooth_fwd}  yaw_ctrl={w_yaw_ctrl}  "
        f"anchor={w_anchor}  anchor_yaw={w_anchor_yaw}  acc_reg={w_acc_reg}  "
        f"orient_gravity={w_orient_gravity}  orient_smooth={w_orient_smooth}  "
        f"wbam={w_wbam}  orient_contact={w_orient_contact}  "
        f"local_contact={w_local_contact}  local_anchor={w_local_anchor}"
    )
    if staged:
        print(f"[INFO] Staged: phase-1 state-only ({s1_iters} iters) → phase-2 full")

    # ── Loss logs ─────────────────────────────────────────────────────
    loss_logs: Dict[str, List[float]] = {
        k: [] for k in ["total", "reproj", "contact", "lat", "smooth_fwd",
                         "yaw_ctrl", "anchor", "anchor_yaw", "acc_reg",
                         "ctrl_jerk", "lat_vel", "orient_gravity", "orient_smooth",
                         "orient_yaw_align",
                         "wbam", "orient_contact", "local_contact",
                         "local_anchor", "local_jerk"]
    }

    # ── Optimization loop ──────────────────────────────────────────────
    t_frames = torch.arange(T - 1, dtype=torch.float32, device=device)

    for it in range(args.iters):

        # Phase transition
        if staged and it == s1_iters:
            optimizer = _make_optimizer(phase2=True)
            if use_lr_decay:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, args.iters - s1_iters), eta_min=args.lr_end)
            print(f"[INFO] Phase-2 start at iter {it}: controls now optimized.")

        in_phase1 = staged and (it < s1_iters)
        optimizer.zero_grad()

        # ── Build a_ctrl for this iteration ────────────────────────────
        if use_pulse and not in_phase1:
            a_ctrl_eff = build_a_ctrl_from_pulses(
                pulse_amp, pulse_mean, pulse_logw, a_lat_var,
                sprint_dir_t, T, device,
            )
            yaw_acc_eff = yaw_acc_var
        elif in_phase1:
            # Phase-1: hold a_ctrl fixed at init (no gradient through controls)
            a_ctrl_eff  = a_ctrl_var.detach()
            yaw_acc_eff = yaw_acc_var.detach()
        else:
            a_ctrl_eff  = a_ctrl_var
            yaw_acc_eff = yaw_acc_var

        # ── Rollout ────────────────────────────────────────────────────
        x_seq, v_seq, yaw_seq, _ = rollout_states(
            x0_var, v0_var, yaw0_var, yaw_rate0_var,
            a_ctrl_eff, yaw_acc_eff, dt,
        )  # (T, 3), (T, 3), (T,), (T,)

        # ── Root orientation ────────────────────────────────────────────
        if use_full_dof_orient:
            alpha3d_eff = alpha3d_var if not in_phase1 else alpha3d_var.detach()
            R_root, _ = rollout_orientation_full_dof(R0_full_dof, w3d_0_var, alpha3d_eff, dt)
        else:
            R_root = combine_yaw_with_sam3d(yaw_seq, R_sam_world)  # (T, 3, 3)
        go_world_t = rotmat_to_rotvec_diff(R_root)                 # (T, 3)

        # ── Local rotation rollout (optional) ─────────────────────────
        if use_local_rot:
            alpha_key_eff  = alpha_key_var if not in_phase1 else alpha_key_var.detach()
            theta_key_seq  = rollout_local_rot(theta0_key_var, omega0_key_var,
                                               alpha_key_eff, dt)   # (T, n_key, 3)
            body_pose_eff  = build_body_pose_opt(body_pose_t, theta_key_seq)
        else:
            body_pose_eff  = body_pose_t
            theta_key_seq  = None

        # ── SMPL forward pass ──────────────────────────────────────────
        J_world = run_smpl(smpl_model, go_world_t, body_pose_eff,
                           betas_t, x_seq, device)             # (T, 24, 3)

        # J_world with global_orient detached — used for WBAM so that
        # angular-momentum gradients only flow through transl + local rots,
        # not through global_orient (prevents WBAM competing with gravity).
        if w_wbam > 0 and not in_phase1:
            J_world_wbam = run_smpl(smpl_model, go_world_t.detach(), body_pose_eff,
                                    betas_t, x_seq, device)
        else:
            J_world_wbam = J_world

        # ── Contact label update (periodically, no gradient) ──────────
        _do_contact_update = (
            (w_wbam > 0 or w_orient_contact > 0 or w_local_contact > 0)
            and not in_phase1
            and it % biom_update_iters == 0
        )
        if _do_contact_update:
            _J_np   = J_world.detach().cpu().numpy()
            _wu_np  = world_up.detach().cpu().numpy()
            if args.use_contact_loss:
                _floor = float(floor_y_var.item())
            else:
                # Project foot positions onto world_up (height along vertical axis)
                _foot_pts = _J_np[:, _FOOT_JOINT_IDS, :]
                _foot_h   = (_foot_pts * _wu_np[None, None, :]).sum(-1).min(axis=1)
                _floor    = float(_foot_h.min()) - 0.05
            phi_np = compute_contact_labels_np(
                _J_np, _floor,
                foot_ids=_FOOT_JOINT_IDS, margin=biom_margin,
                world_up_vec=_wu_np,
            )
            phi_t = torch.tensor(phi_np, device=device, dtype=torch.float32)

        # ── Camera (shared static or identity) ────────────────────────
        if args.opt_cam:
            R_cam_t    = rotvec_to_matrix(cam_rvec_var).expand(T, 3, 3)
            cam_tvec_t = cam_tvec_var.expand(T, 3)
        else:
            R_cam_t    = torch.eye(3, device=device).unsqueeze(0).expand(T, 3, 3)
            cam_tvec_t = torch.zeros(T, 3, device=device)

        # ── Reprojection loss ──────────────────────────────────────────
        uv_pred  = project_joints(J_world, R_cam_t, cam_tvec_t, focal_t, cx, cy)
        L_reproj = loss_reprojection(uv_pred, obs2d_t, args.huber_delta,
                                     frame_weights=obs2d_conf_t)

        # ── Contact loss (phase-2 only) ────────────────────────────────
        L_contact = x_seq.new_zeros(1).squeeze()
        if w_contact > 0 and not in_phase1:
            L_contact = w_contact * loss_ground_contact(
                J_world, floor_y_var,
                foot_ids=_FOOT_JOINT_IDS,
                margin=args.contact_margin,
                w_penetration=args.contact_w_pen,
                w_liftoff=args.contact_w_lift,
            )

        # ── Control prior losses (phase-2 only) ───────────────────────
        L_lat       = x_seq.new_zeros(1).squeeze()
        L_smooth    = x_seq.new_zeros(1).squeeze()
        L_yaw_ctrl  = x_seq.new_zeros(1).squeeze()
        L_acc_reg   = x_seq.new_zeros(1).squeeze()
        L_ctrl_jerk = x_seq.new_zeros(1).squeeze()
        L_lat_vel   = x_seq.new_zeros(1).squeeze()
        if not in_phase1:
            if w_lat > 0:
                L_lat = w_lat * loss_control_lateral(
                    a_ctrl_eff, sprint_dir_t, normalized=normalize_lat)
            if w_smooth_fwd > 0:
                L_smooth = w_smooth_fwd * loss_control_smooth_fwd(a_ctrl_eff, sprint_dir_t)
            if w_yaw_ctrl > 0:
                L_yaw_ctrl = w_yaw_ctrl * loss_control_yaw(yaw_acc_eff)
            if w_acc_reg > 0:
                L_acc_reg = w_acc_reg * (a_ctrl_eff ** 2).sum(dim=-1).mean()
            if w_ctrl_jerk > 0:
                L_ctrl_jerk = w_ctrl_jerk * loss_ctrl_jerk(a_ctrl_eff)
            if w_lat_vel > 0:
                L_lat_vel = w_lat_vel * loss_lateral_velocity(
                    v_seq, sprint_dir_t, normalized=normalize_lat)

        # ── Full-DOF orientation losses (phase-2 only) ─────────────────
        L_orient_gravity   = x_seq.new_zeros(1).squeeze()
        L_orient_smooth    = x_seq.new_zeros(1).squeeze()
        L_orient_yaw_align = x_seq.new_zeros(1).squeeze()
        w_orient_yaw_align = getattr(args, 'w_orient_yaw_align', 0.0)
        if not in_phase1 and use_full_dof_orient:
            if w_orient_gravity > 0:
                L_orient_gravity = w_orient_gravity * loss_orient_gravity_align(R_root, world_up)
            if w_orient_smooth > 0 and alpha3d_eff is not None:
                L_orient_smooth = w_orient_smooth * loss_orient_smooth(alpha3d_eff)
            if w_orient_yaw_align > 0:
                L_orient_yaw_align = w_orient_yaw_align * loss_orient_yaw_align(
                    R_root, sprint_dir_t, world_up)

        # ── Biomechanics losses (phase-2 only) ─────────────────────────
        L_wbam           = x_seq.new_zeros(1).squeeze()
        L_orient_contact = x_seq.new_zeros(1).squeeze()
        L_local_contact  = x_seq.new_zeros(1).squeeze()
        L_local_anchor   = x_seq.new_zeros(1).squeeze()
        L_local_jerk     = x_seq.new_zeros(1).squeeze()
        if not in_phase1:
            _phi = phi_t
            if w_wbam > 0:
                L_wbam = w_wbam * loss_wbam(J_world_wbam, dt, phi_t=_phi)
            if w_orient_contact > 0 and use_full_dof_orient and alpha3d_eff is not None:
                _phi_alpha = _phi[:-1] if (_phi is not None and _phi.shape[0] == T) else _phi
                L_orient_contact = w_orient_contact * loss_orient_contact(
                    alpha3d_eff, _phi_alpha if _phi_alpha is not None else torch.ones(T-1, device=device),
                )
            if use_local_rot and alpha_key_var is not None:
                alpha_key_eff2 = alpha_key_var if not in_phase1 else alpha_key_var.detach()
                _phi_a = _phi[:-1] if (_phi is not None and _phi.shape[0] == T) else \
                         (_phi if _phi is not None else torch.ones(T-1, device=device))
                if w_local_contact > 0:
                    L_local_contact = w_local_contact * loss_local_contact(alpha_key_eff2, _phi_a)
                if w_local_anchor > 0 and theta_key_seq is not None:
                    L_local_anchor = w_local_anchor * loss_local_anchor(
                        theta_key_seq, body_pose_key_anchor_t)
                if w_local_jerk > 0:
                    if alpha_key_eff2.shape[0] >= 2:
                        da_k = alpha_key_eff2[1:] - alpha_key_eff2[:-1]
                        L_local_jerk = w_local_jerk * (da_k ** 2).sum(dim=-1).sum(dim=-1).mean()

        # ── Anchor losses ──────────────────────────────────────────────
        L_anchor     = x_seq.new_zeros(1).squeeze()
        L_anchor_yaw = x_seq.new_zeros(1).squeeze()
        if w_anchor > 0:
            L_anchor = w_anchor * loss_anchor_translation(
                x_seq, x_anchor_t, frame_weights=obs2d_conf_t)
        if w_anchor_yaw > 0:
            L_anchor_yaw = w_anchor_yaw * loss_anchor_yaw(
                yaw_seq, yaw_sam_t, frame_weights=obs2d_conf_t)

        # ── Total ──────────────────────────────────────────────────────
        w_repr_eff = args.w_reproj_stage1 if in_phase1 else w_reproj
        total = (
            w_repr_eff    * L_reproj
            + L_contact
            + L_lat
            + L_smooth
            + L_yaw_ctrl
            + L_anchor
            + L_anchor_yaw
            + L_acc_reg
            + L_ctrl_jerk
            + L_lat_vel
            + L_orient_gravity
            + L_orient_smooth
            + L_orient_yaw_align
            + L_wbam
            + L_orient_contact
            + L_local_contact
            + L_local_anchor
            + L_local_jerk
        )

        total.backward()
        if args.grad_clip > 0:
            if use_full_dof_orient:
                all_params = [x0_var, v0_var, w3d_0_var]
            else:
                all_params = [x0_var, v0_var, yaw0_var, yaw_rate0_var]
            if use_local_rot:
                all_params = all_params + [theta0_key_var, omega0_key_var]
            if not in_phase1:
                if use_pulse:
                    all_params += [pulse_amp, pulse_mean, pulse_logw, a_lat_var]
                else:
                    all_params += [a_ctrl_var, yaw_acc_var]
                if use_full_dof_orient:
                    all_params += [alpha3d_var]
                if use_local_rot:
                    all_params += [alpha_key_var]
            if args.opt_cam:
                all_params += [cam_rvec_var, cam_tvec_var]
            torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # ── Logging ───────────────────────────────────────────────────
        loss_logs["total"].append(total.item())
        loss_logs["reproj"].append(L_reproj.item())
        loss_logs["contact"].append(L_contact.item() if not isinstance(L_contact, float) else L_contact)
        loss_logs["lat"].append(L_lat.item())
        loss_logs["smooth_fwd"].append(L_smooth.item())
        loss_logs["yaw_ctrl"].append(L_yaw_ctrl.item())
        loss_logs["anchor"].append(L_anchor.item())
        loss_logs["anchor_yaw"].append(L_anchor_yaw.item())
        loss_logs["acc_reg"].append(L_acc_reg.item())
        loss_logs["ctrl_jerk"].append(L_ctrl_jerk.item())
        loss_logs["lat_vel"].append(L_lat_vel.item())
        loss_logs["orient_gravity"].append(L_orient_gravity.item())
        loss_logs["orient_smooth"].append(L_orient_smooth.item())
        loss_logs["orient_yaw_align"].append(L_orient_yaw_align.item())
        loss_logs["wbam"].append(L_wbam.item())
        loss_logs["orient_contact"].append(L_orient_contact.item())
        loss_logs["local_contact"].append(L_local_contact.item())
        loss_logs["local_anchor"].append(L_local_anchor.item())
        loss_logs["local_jerk"].append(L_local_jerk.item())

        if args.debug_every > 0 and (it % args.debug_every == 0 or it == args.iters - 1):
            print(
                f"  iter {it:4d}  total={total.item():.4f}  "
                f"reproj={L_reproj.item():.4f}  lat={L_lat.item():.4f}  "
                f"anchor={L_anchor.item():.4f}  smooth={L_smooth.item():.4f}"
            )
            with torch.no_grad():
                err = (uv_pred - obs2d_t).norm(dim=-1)
                print(f"           reproj: mean={err.mean().item():.2f}px  "
                      f"p90={torch.quantile(err.reshape(-1), 0.9).item():.2f}px")

    # ── Extract final results ──────────────────────────────────────────
    with torch.no_grad():
        if use_pulse:
            a_ctrl_final = build_a_ctrl_from_pulses(
                pulse_amp, pulse_mean, pulse_logw, a_lat_var,
                sprint_dir_t, T, device,
            )
        else:
            a_ctrl_final = a_ctrl_var

        x_seq_f, v_seq_f, yaw_seq_f, _ = rollout_states(
            x0_var, v0_var, yaw0_var, yaw_rate0_var,
            a_ctrl_final, yaw_acc_var, dt,
        )
        if use_full_dof_orient:
            R_root_f, _ = rollout_orientation_full_dof(R0_full_dof, w3d_0_var, alpha3d_var, dt)
        else:
            R_root_f = combine_yaw_with_sam3d(yaw_seq_f, R_sam_world)
        go_world_f = rotmat_to_rotvec_diff(R_root_f)

        if use_local_rot:
            theta_key_f   = rollout_local_rot(theta0_key_var, omega0_key_var, alpha_key_var, dt)
            body_pose_f   = build_body_pose_opt(body_pose_t, theta_key_f)
        else:
            body_pose_f   = body_pose_t
            theta_key_f   = None

        J_world_f  = run_smpl(smpl_model, go_world_f, body_pose_f,
                               betas_t, x_seq_f, device)

        if args.opt_cam:
            R_cam_f    = rotvec_to_matrix(cam_rvec_var).expand(T, 3, 3)
            cam_tvec_f = cam_tvec_var.expand(T, 3)
        else:
            R_cam_f    = torch.eye(3, device=device).unsqueeze(0).expand(T, 3, 3)
            cam_tvec_f = torch.zeros(T, 3, device=device)

        uv_f  = project_joints(J_world_f, R_cam_f, cam_tvec_f, focal_t, cx, cy)
        reproj_err = (uv_f - obs2d_t).norm(dim=-1).cpu().numpy()

    transl_world_np = x_seq_f.cpu().numpy()
    go_world_np     = go_world_f.cpu().numpy()
    cam_rvec_out    = cam_rvec_var.expand(T, 3).cpu().numpy()
    cam_tvec_out    = cam_tvec_var.expand(T, 3).cpu().numpy()
    cam_R_out       = R_cam_f.cpu().numpy()

    pulse_info = None
    if use_pulse:
        pulse_info = {
            "amplitudes": pulse_amp.detach().cpu().numpy(),
            "means":      pulse_mean.detach().cpu().numpy(),
            "sigmas":     torch.exp(pulse_logw).detach().cpu().numpy(),
            "K":          K,
        }

    body_pose_opt_np  = body_pose_f.cpu().numpy()
    contact_labels_np = phi_np if phi_np is not None else np.zeros(T, dtype=np.float32)
    joints_world_np   = J_world_f.cpu().numpy()   # (T, 24, 3)

    return {
        "transl_world":    transl_world_np,
        "go_world":        go_world_np,
        "body_pose_opt":   body_pose_opt_np,   # (T, 69) — optimized if local_rot, else SAM3D
        "joints_world":    joints_world_np,    # (T, 24, 3) — 3D joints for WBAM analysis
        "contact_labels":  contact_labels_np,  # (T,) stance=1 flight=0
        "v_seq":           v_seq_f.cpu().numpy(),
        "yaw_seq":         yaw_seq_f.cpu().numpy(),
        "a_ctrl":          a_ctrl_final.detach().cpu().numpy(),
        "cam_rvec":        cam_rvec_out,
        "cam_tvec":        cam_tvec_out,
        "cam_R":           cam_R_out,
        "focal":           focal_np.copy(),
        "floor_y":         float(floor_y_var.item()),
        "sprint_dir":      sprint_dir_np,
        "pulse_info":      pulse_info,
        "loss_logs":       loss_logs,
        "reproj_err_mean": float(reproj_err.mean()),
        "reproj_err_p90":  float(np.percentile(reproj_err, 90)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Section 8: Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Physics-guided rollout-based global human motion reconstruction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── I/O ────────────────────────────────────────────────────────────────
    io = p.add_argument_group("I/O")
    io.add_argument("--in_pkl",          required=True)
    io.add_argument("--out_pkl",         required=True)
    io.add_argument("--smpl_model_path", required=True)
    io.add_argument("--video",           default="")
    io.add_argument("--gender",          default="neutral", choices=["neutral","male","female"])
    io.add_argument("--device",          default="cpu")
    io.add_argument("--cam_R_key",       default="cam_R")
    io.add_argument("--cam_t_key",       default="cam_t")

    # ── Scene ──────────────────────────────────────────────────────────────
    sc = p.add_argument_group("Scene / camera")
    sc.add_argument("--cx",              type=float, default=0.0)
    sc.add_argument("--cy",              type=float, default=0.0)
    sc.add_argument("--fps",             type=float, default=0.0)
    sc.add_argument("--units",           default="m", choices=["m","cm"])
    sc.add_argument("--target_height_m", type=float, default=1.6)
    sc.add_argument("--opt_cam",         type=int,   default=0,
                    help="0=identity camera (recommended), 1=optimize one shared static camera")
    sc.add_argument("--lr_cam",          type=float, default=1e-2)

    # ── Rollout / model ────────────────────────────────────────────────────
    ro = p.add_argument_group("Rollout model")
    ro.add_argument("--use_pulse_model", type=int, default=0,
                    help="0=raw per-frame a_ctrl, 1=Gaussian pulse parameterization")
    ro.add_argument("--n_pulses",        type=int, default=8,
                    help="Number of Gaussian pulses (only used if use_pulse_model=1). "
                         "Pulse amplitudes can be positive (acceleration) or negative "
                         "(deceleration); initialized via least-squares fit to the "
                         "smoothed finite-difference init, not at zero.")
    ro.add_argument("--orient_mode",     default="yaw_sam_pitchroll",
                    choices=["yaw_sam_pitchroll", "yaw_only", "full_dof"],
                    help="yaw_sam_pitchroll: replace yaw, keep SAM3D pitch/roll (default). "
                         "yaw_only: pure yaw rotation, zero pitch/roll. "
                         "full_dof: integrate full 3D angular acceleration on SO(3) — "
                         "adds w3d_0 (initial angular velocity) and alpha3d (angular "
                         "acceleration controls) variables; yaw0/yaw_acc are unused.")
    ro.add_argument("--w_orient_yaw_align", type=float, default=0.0,
                    help="Yaw alignment: penalises chest_fwd deviating from sprint_dir "
                         "in horizontal plane. Separate from gravity (pitch/roll). full_dof only.")
    ro.add_argument("--w_orient_gravity", type=float, default=0.0,
                    help="Gravity alignment loss weight: spine_up should align with world up. "
                         "For full_dof orient only. Replaces w_anchor_yaw. "
                         "Typical range: 0.5-2.0")
    ro.add_argument("--w_orient_smooth", type=float, default=0.0,
                    help="3D angular acceleration regularizer for full_dof orient. "
                         "Replaces w_yaw_ctrl. Typical range: 0.01-0.1")
    ro.add_argument("--sprint_dir_mode", default="sam3d_orient",
                    choices=["sam3d_orient", "pca_traj"],
                    help="How to estimate the sprint direction vector d̂:\n"
                         "  sam3d_orient (default): circular mean of SAM3D body forward "
                         "  direction projected to XZ. More stable — estimated directly "
                         "  from 2D pose keypoints, not from noisy depth-scaled trajectory.\n"
                         "  pca_traj: PCA on the initial XZ trajectory positions (legacy).")
    ro.add_argument("--sprint_plane",    default="xz", choices=["xz","xyz"])

    # ── Loss weights ───────────────────────────────────────────────────────
    lw = p.add_argument_group("Loss weights")
    lw.add_argument("--w_reproj",        type=float, default=1.0)
    lw.add_argument("--w_reproj_stage1", type=float, default=5.0)
    lw.add_argument("--w_lat",           type=float, default=1.0,
                    help="Cross-track (lateral) acceleration penalty weight. "
                         "With --normalize_lat_loss=1 (default), the raw loss is in [0,1] "
                         "so w_lat=1 means lateral energy should not exceed total reproj loss. "
                         "Typical range: 0.5-5.0")
    lw.add_argument("--normalize_lat_loss", type=int, default=1,
                    help="1=scale-invariant ratio loss (recommended), 0=raw squared magnitude. "
                         "The normalized form is always in [0,1] regardless of a_ctrl magnitude.")
    lw.add_argument("--w_smooth_fwd",    type=float, default=0.05,
                    help="Forward acceleration smoothness penalty weight")
    lw.add_argument("--w_yaw_ctrl",      type=float, default=0.05,
                    help="Yaw angular acceleration regularizer weight")
    lw.add_argument("--w_anchor",        type=float, default=0.05,
                    help="Weak translation anchor to SAM3D/HS init (prevents drift)")
    lw.add_argument("--w_anchor_yaw",    type=float, default=0.01,
                    help="Weak yaw anchor to SAM3D yaw estimates")
    lw.add_argument("--w_acc_reg",       type=float, default=0.0,
                    help="Soft L2 penalty on acceleration magnitude (prevents runaway controls). "
                         "Typical range: 0.0-0.01")
    lw.add_argument("--w_ctrl_jerk",    type=float, default=0.0,
                    help="Full 3-D control jerk penalty: mean||a_{t+1}-a_t||². "
                         "Penalizes lateral acceleration oscillations that w_smooth_fwd misses. "
                         "Directly reduces the jerk metric. Typical range: 0.01-0.5")
    lw.add_argument("--w_lat_vel",      type=float, default=0.0,
                    help="Lateral velocity penalty: mean(||v_perp||²/||v||²). "
                         "Directly suppresses cross-track velocity (fixes sprint_align and v_lat). "
                         "Unlike w_lat (acceleration), this also corrects the initial v0. "
                         "Typical range: 0.5-5.0")
    lw.add_argument("--smooth_a_ctrl_sigma", type=float, default=3.0,
                    help="Gaussian sigma (frames) for trajectory pre-smoothing before computing "
                         "a_ctrl_init. Removes per-frame depth jitter that creates ~100-1000 m/s² "
                         "spurious accelerations. 0=disabled. 3.0≈0.1s at 30fps (recommended).")
    lw.add_argument("--huber_delta",     type=float, default=15.0)

    # ── Confidence weighting ───────────────────────────────────────────────────
    cf = p.add_argument_group("Confidence weighting")
    cf.add_argument("--use_conf_weights", type=int, default=0,
                    help="1=weight reprojection and anchor losses by per-frame pose fit "
                         "confidence (derived from 'fit_errors' in the input pkl). "
                         "Frames where SAM3D fitted poorly are down-weighted so their "
                         "noise is rejected from the optimization.")
    cf.add_argument("--conf_sigma",       type=float, default=1.0,
                    help="Softness of the confidence curve: w=exp(-err/(sigma*median)). "
                         "Larger sigma → gentler contrast between good and bad frames. "
                         "Typical range: 0.5–2.0")
    cf.add_argument("--conf_min",         type=float, default=0.1,
                    help="Minimum confidence weight floor so no frame is zeroed out. "
                         "Typical range: 0.05–0.2")

    # ── Contact loss ───────────────────────────────────────────────────────
    ct = p.add_argument_group("Contact loss")
    ct.add_argument("--use_contact_loss",type=int,   default=0)
    ct.add_argument("--w_contact",       type=float, default=1.0)
    ct.add_argument("--contact_margin",  type=float, default=0.05)
    ct.add_argument("--contact_w_pen",   type=float, default=1.0)
    ct.add_argument("--contact_w_lift",  type=float, default=0.5)

    # ── Biomechanics losses ────────────────────────────────────────────────
    bm = p.add_argument_group("Biomechanics losses (contact-conditioned angular momentum)")
    bm.add_argument("--w_wbam",           type=float, default=0.0,
                    help="Whole-body angular momentum near-zero loss weight. "
                         "Enforces Herr & Popov (2008) angular momentum regulation. "
                         "Gradient flows through global_orient + local joint positions. "
                         "Typical: 0.1-0.5. full_dof orient recommended.")
    bm.add_argument("--w_orient_contact", type=float, default=0.0,
                    help="Contact-conditioned global orient smoothness (replaces w_orient_smooth "
                         "when > 0). Stance frames: w_stance=1.0 tighter; flight: w_flight=0.1 "
                         "looser. Typical: 0.05.")
    bm.add_argument("--w_local_contact",  type=float, default=0.0,
                    help="Contact-conditioned local joint angular acceleration smoothness. "
                         "Applied to key joints (hips, spine, knees, shoulders). Typical: 0.05.")
    bm.add_argument("--w_local_anchor",   type=float, default=2.0,
                    help="L2 anchor weight pulling local joint angles toward SAM3D observations. "
                         "Strong anchor prevents implausible poses. Typical: 1.0-3.0.")
    bm.add_argument("--w_local_jerk",     type=float, default=0.05,
                    help="Local joint angular jerk penalty. Typical: 0.01-0.1.")
    bm.add_argument("--local_rot_mode",   default="fixed",
                    choices=["fixed", "velocity_rollout"],
                    help="fixed: body_pose fixed from SAM3D (default). "
                         "velocity_rollout: optimize key joint angles (hips, spine, knees, "
                         "shoulders) via angular velocity rollout — couples WBAM gradient "
                         "through local rotations.")
    bm.add_argument("--local_rot_lr_scale", type=float, default=0.3,
                    help="LR multiplier for local rot alpha_key_var (relative to global lr). "
                         "Keep small (0.1-0.5) to prevent large deviations from SAM3D.")
    bm.add_argument("--biom_margin",      type=float, default=0.08,
                    help="Foot-floor contact detection margin in metres for phi computation.")
    bm.add_argument("--biom_update_iters",type=int,   default=50,
                    help="Re-detect contact labels every N optimization iterations.")

    # ── Optimization ───────────────────────────────────────────────────────
    op = p.add_argument_group("Optimization hyperparameters")
    op.add_argument("--iters",           type=int,   default=1000)
    op.add_argument("--lr",              type=float, default=3e-2)
    op.add_argument("--lr_end",          type=float, default=1e-4)
    op.add_argument("--stage1_iters",    type=int,   default=300,
                    help="Iterations of phase-1 (initial state only, controls frozen)")
    op.add_argument("--grad_clip",       type=float, default=1.0)
    op.add_argument("--debug_every",     type=int,   default=200)

    # ── Pose init ──────────────────────────────────────────────────────────
    hc = p.add_argument_group("Pose initialization")
    hc.add_argument("--pelvis_id",      type=int,   default=0,
                    help="SMPL joint index used as the pelvis root (default 0).")

    # ── PnP ────────────────────────────────────────────────────────────────
    pnp = p.add_argument_group("PnP camera initialization")
    pnp.add_argument("--use_pnp_init",  type=int,   default=0,
                     help="0=use SAM3D camera directly (recommended for rollout)")

    # ── Output ─────────────────────────────────────────────────────────────
    out = p.add_argument_group("Output")
    out.add_argument("--store_per_iter_stats", type=int, default=1)

    return p


# ══════════════════════════════════════════════════════════════════════════════
# Section 9: Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = build_parser()
    args   = parser.parse_args()

    if smplx is None:
        print("[ERROR] smplx required. Install: pip install smplx"); sys.exit(1)

    device = torch.device(args.device)
    print(f"[INFO] rollout_motion_optim.py  device={device}")

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"[INFO] Loading: {args.in_pkl}")
    data = load_pkl(args.in_pkl)
    sp   = data.get("smpl_parameters", {})

    body_pose_raw = sp.get("body_pose")
    betas_raw     = sp.get("betas")
    go_cam_raw    = sp.get("global_orient")
    transl_raw    = sp.get("transl_final")
    # focal_length lives inside smpl_parameters; fall back to top-level for older pkls
    focal_raw     = sp.get("focal_length") if sp.get("focal_length") is not None else data.get("focal_length")
    obs2d_raw     = data.get("smpl_kpts2d_24")

    for name, val in [
        ("body_pose", body_pose_raw), ("betas", betas_raw),
        ("global_orient", go_cam_raw), ("transl_final", transl_raw),
        ("focal_length", focal_raw),  ("smpl_kpts2d_24", obs2d_raw),
    ]:
        if val is None:
            print(f"[ERROR] Missing key: {name}"); sys.exit(1)

    T = np.asarray(body_pose_raw).reshape(-1, 69).shape[0]
    print(f"[INFO] Sequence length T={T}")

    body_pose_np = ensure_TxD(body_pose_raw, T, 69, "body_pose")
    betas_np     = np.asarray(betas_raw, dtype=np.float32)
    if betas_np.ndim == 1:
        betas_np = betas_np[None].repeat(T, axis=0)
    elif betas_np.shape[0] == 1:
        betas_np = np.repeat(betas_np, T, axis=0)
    B = betas_np.shape[1]
    betas_np     = ensure_TxD(betas_np, T, B, "betas")
    go_cam_np    = ensure_TxD(go_cam_raw, T, 3, "global_orient")
    transl_cam_np = ensure_TxD(transl_raw, T, 3, "transl")
    focal_np     = ensure_T(focal_raw, T, "focal_length")
    obs2d_np     = np.asarray(obs2d_raw, dtype=np.float32)
    if obs2d_np.shape != (T, 24, 2):
        raise ValueError(f"smpl_kpts2d_24: expected ({T},24,2), got {obs2d_np.shape}")

    if args.units == "cm":
        transl_cam_np *= 0.01

    # ── Camera intrinsics ──────────────────────────────────────────────────
    # cx/cy MUST match the coordinate system of obs2d (smpl_kpts2d_24).
    # The in_pkl stores the exact cx/cy used when projecting smpl_kpts2d_24
    # under camera_intrinsics_used — use that as the ground truth, NOT the
    # video resolution (which would introduce a phantom X-shift of ~W/2 metres
    # into transl_world and cause ~20-30° global_orient drift).
    cam_ci = data.get("camera_intrinsics_used", {})
    cx_obs = cam_ci.get("cx", None)   # None = not recorded in pkl
    cy_obs = cam_ci.get("cy", None)

    cx, cy, fps = args.cx, args.cy, args.fps
    _cx_determined = False   # track if cx was set from a reliable source
    if args.video and os.path.isfile(args.video):
        cx_v, cy_v, fps_v, _, _ = get_video_info(args.video)
        # Only pull fps from video; cx/cy come from the obs2d intrinsics.
        if fps == 0.0: fps = fps_v
        print(f"[INFO] Video fps={fps:.2f}  (cx/cy taken from obs2d intrinsics, not video)")
    # Override cx/cy with obs2d intrinsics unless explicitly set by the caller.
    if cx == 0.0 and cx_obs is not None:
        cx = float(cx_obs)
        _cx_determined = True
    if cy == 0.0 and cy_obs is not None:
        cy = float(cy_obs)
        _cx_determined = True
    print(f"[INFO] Reprojection intrinsics: cx={cx:.1f} cy={cy:.1f}")
    # Last-resort fallback: only if no source provided cx (not when cx=0 is intentional)
    if not _cx_determined and cx == 0.0 or (not _cx_determined and cy == 0.0):
        cx = float(obs2d_np[:, :, 0].mean())
        cy = float(obs2d_np[:, :, 1].mean())
    if fps == 0.0: fps = 30.0
    dt = 1.0 / fps

    # ── SMPL model ─────────────────────────────────────────────────────────
    print(f"[INFO] Loading SMPL from: {args.smpl_model_path}")
    smpl_model = smplx.create(
        args.smpl_model_path, model_type="smpl",
        gender=args.gender, use_pca=False, batch_size=T,
    ).to(device)
    smpl_model.eval()
    for param in smpl_model.parameters():
        param.requires_grad_(False)

    # ── SAM3D camera poses → world frame ───────────────────────────────────
    cam_R_raw, cam_t_raw = load_sam3d_camera_poses(
        data, cam_R_key=args.cam_R_key, cam_t_key=args.cam_t_key)

    if cam_R_raw is not None and cam_t_raw is not None:
        if cam_R_raw.shape[0] != T:
            cam_R_raw = cam_R_raw[:T]
            cam_t_raw = cam_t_raw[:T]
        R_cam0, t_cam0 = build_world_frame_cameras(cam_R_raw, cam_t_raw)
    else:
        R_cam0 = np.tile(np.eye(3, dtype=np.float32)[None], (T, 1, 1))
        t_cam0 = np.zeros((T, 3), dtype=np.float32)

    # ── Human world-frame pose init (SAM3D transl directly as p_cam) ──────
    # SAM3D's transl_final is the camera-frame pelvis estimate — use it
    # directly instead of re-deriving depth from the pixel height formula.
    # This gives a smoother init (SAM3D already joint-fits pose + position)
    # and avoids the stride-frequency Z oscillations of height-scale.
    p_cam_t = transl_cam_np.copy()   # (T, 3)  pelvis in camera frame (metres)
    print(f"[INFO] SAM3D pelvis depth: mean={p_cam_t[:,2].mean():.2f} "
          f"min={p_cam_t[:,2].min():.2f} max={p_cam_t[:,2].max():.2f} m")

    transl_world_init, go_world_init = init_human_world_pose(
        p_cam_t, R_cam0, t_cam0, go_cam_np,
        smpl_model, body_pose_np, betas_np,
        pelvis_id=args.pelvis_id, device=device,
    )
    print(f"[INFO] Transl world init: mean={transl_world_init.mean(0).round(3)}")

    # Camera init (shared static: use mean of SAM3D poses)
    cam_rvec_init = matrix_to_rotvec_np(R_cam0)
    cam_tvec_init = t_cam0.copy()

    # ── Sprint direction ───────────────────────────────────────────────────
    if args.sprint_dir_mode == "sam3d_orient":
        # Preferred: use SAM3D global orientation's forward direction (XZ plane).
        # More stable than PCA because it's estimated from 2D body keypoints
        # rather than the noisy depth-scaled trajectory.
        sprint_dir_np = estimate_sprint_dir_from_sam3d(go_world_init)
    else:
        # Fallback: PCA on the initial XZ trajectory positions
        sprint_dir_np = estimate_sprint_direction(transl_world_init, plane=args.sprint_plane)
        print(f"[INFO] Sprint direction from traj PCA (plane={args.sprint_plane}): "
              f"d={sprint_dir_np.round(4)}")

    # ── Confidence weights from per-frame pose fit quality ─────────────────
    obs2d_conf_np = None
    if args.use_conf_weights:
        fit_errors_raw = data.get("fit_errors", None)
        if fit_errors_raw is not None:
            obs2d_conf_np = compute_conf_weights(
                np.asarray(fit_errors_raw, dtype=np.float32),
                sigma=args.conf_sigma,
                min_weight=args.conf_min,
            )
            print(f"[INFO] Confidence weights from fit_errors: "
                  f"min={obs2d_conf_np.min():.3f}  max={obs2d_conf_np.max():.3f}  "
                  f"mean={obs2d_conf_np.mean():.3f}  "
                  f"low-conf frames (<0.5): {int((obs2d_conf_np < 0.5).sum())}/{T}")
        else:
            print("[WARN] --use_conf_weights=1 but 'fit_errors' not found in pkl "
                  "— running with uniform weights")

    # ── Run rollout optimization ───────────────────────────────────────────
    t0 = time.time()
    results = run_rollout_optimization(
        smpl_model=smpl_model,
        body_pose_np=body_pose_np,
        betas_np=betas_np,
        obs2d_np=obs2d_np,
        focal_np=focal_np,
        cx=cx, cy=cy, dt=dt,
        transl_world_init=transl_world_init,
        go_world_init=go_world_init,
        cam_rvec_init=cam_rvec_init,
        cam_tvec_init=cam_tvec_init,
        sprint_dir_np=sprint_dir_np,
        args=args,
        device=device,
        obs2d_conf_np=obs2d_conf_np,
    )
    print(f"[INFO] Done in {time.time()-t0:.1f}s  "
          f"reproj mean={results['reproj_err_mean']:.2f}px  "
          f"p90={results['reproj_err_p90']:.2f}px")

    # ── Build output PKL ───────────────────────────────────────────────────
    out = copy.deepcopy(data)
    out["smpl_parameters"]["global_orient_world_refined"] = results["go_world"]
    out["smpl_parameters"]["transl_world_refined"]        = results["transl_world"]
    out["smpl_parameters"]["focal_length_refined"]        = results["focal"]

    loss_logs_out = results["loss_logs"] if args.store_per_iter_stats else {
        k: [v[-1]] if v else [] for k, v in results["loss_logs"].items()
    }

    # Store in joint_cam_human_optim format so existing eval scripts work
    out["joint_cam_human_optim"] = {
        "transl_world":        results["transl_world"],
        "global_orient_world": results["go_world"],
        "body_pose_opt":       results["body_pose_opt"],   # (T,69) optimized local rots
        "joints_world":        results["joints_world"],    # (T,24,3) for WBAM analysis
        "contact_labels":      results["contact_labels"],  # (T,) stance/flight
        "cam_rvec":            results["cam_rvec"],
        "cam_tvec":            results["cam_tvec"],
        "cam_R":               results["cam_R"],
        "focal":               results["focal"],
        "cx": cx, "cy": cy, "fps": fps,
        "loss_logs":           loss_logs_out,
        "sprint_dir":          sprint_dir_np,
        "rollout_info": {
            "v_seq":       results["v_seq"],
            "yaw_seq":     results["yaw_seq"],
            "a_ctrl":      results["a_ctrl"],
            "use_pulse":   args.use_pulse_model,
            "pulse_info":  results["pulse_info"],
        },
        "viz_stats": {
            "reproj_err_mean_px": results["reproj_err_mean"],
            "reproj_err_p90_px":  results["reproj_err_p90"],
            "target_height_m":    args.target_height_m,
        },
        "weights": {
            "w_reproj":    args.w_reproj,
            "w_lat":       args.w_lat,
            "w_smooth_fwd":args.w_smooth_fwd,
            "w_yaw_ctrl":  args.w_yaw_ctrl,
            "w_anchor":    args.w_anchor,
        },
    }

    save_pkl(out, args.out_pkl)
    print(f"[saved] {args.out_pkl}")


if __name__ == "__main__":
    main()


# ── Example ───────────────────────────────────────────────────────────────────
# python core/rollout_motion_optim.py \
#     --in_pkl  $SAM3D_DIR/outputs/smpl_sequences/Goree/smpl_sequence.pkl \
#     --out_pkl outputs/rollout/rollout_full.pkl \
#     --smpl_model_path /home/nan/Desktop/NRMFOptim/smplhub/smpl/SMPL_N_model_generate_from_npz.pkl \
#     --video   /home/nan/Desktop/NRMFOptim/sprintMesh2/sprint_videos/original.mp4 \
#     --device  cuda \
#     --iters 1000 --stage1_iters 300 \
#     --w_lat 0.5 --w_smooth_fwd 0.1 --w_anchor 0.05
