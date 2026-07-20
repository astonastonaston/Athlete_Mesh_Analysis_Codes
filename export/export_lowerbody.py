#!/usr/bin/env python3
"""
export_lowerbody.py — Export two sprint keypoint datasets per athlete.

Reads:
  rollout_results/{Athlete}/rollout.pkl     — SMPL-24 optimised world positions
  rollout_results/{Athlete}/mhr_kpts.pkl   — MHR 127-joint skeleton + bone rotations

Outputs:
  sprint_smpl_dataset/data/{Athlete}.json       (Dataset v1 — SMPL-24)
  sprint_lowerbody_dataset/data/{Athlete}.json  (Dataset v2 — MHR lower-body)

─── Dataset v1  sprint_smpl_dataset ─────────────────────────────────────────
  All 24 standard SMPL joints, per-frame 2-D pixel coordinates only.
  Also includes per-frame body orientation vectors (spine_up, chest_fwd,
  body_right) and scalar pose angles.
  Coordinate system: Y-DOWN camera frame (world = camera, opt_cam=0).

─── Dataset v2  sprint_lowerbody_dataset ────────────────────────────────────
  21 lower-body joints from the 127-joint MHR skeleton, plus 3 derived foot
  landmarks (heel_tip, heel_contact, toe_tip) per foot computed from bone
  rotation matrices.  3-D world positions (Y-up, floor=0) + 2-D pixels.

  Derived landmark definitions (left side; right is symmetric):
    heel_tip(t)     = l_foot + R_foot @ POSTERIOR * 5.5 cm
    heel_contact(t) = l_foot + R_foot @ POSTERIOR * 5.0 cm
                              + R_foot @ INFERIOR  * 11.0 cm
    toe_tip(t)      = l_ball + R_ball @ DISTAL     * 3.5 cm

Run:
  conda activate sam_3d_body
  cd /path/to/Athlete_Mesh_Analysis_Codes
  python export/export_lowerbody.py               # all athletes
  python export/export_lowerbody.py --athlete Goree
"""
import argparse, json, os, pickle
import numpy as np

HERE        = os.path.dirname(os.path.abspath(__file__))
ROLLOUT_DIR = os.path.join(HERE, "..", "..", "outputs_with_moge2", "rollout_results")
OUT_V1      = os.path.join(HERE, "..", "..", "sprint_smpl_dataset",      "data")
OUT_V2      = os.path.join(HERE, "..", "..", "sprint_lowerbody_dataset", "data")
ATHLETES    = ["Bishop", "Colin_Brazzell", "Goree", "Jackson", "Original", "Poteat", "Walton"]

W, H = 1920, 1080

# ── SMPL-24 (v1) ──────────────────────────────────────────────────────────────

SMPL24_NAMES = [
    "pelvis",
    "left_hip",  "right_hip",  "spine1",
    "left_knee", "right_knee", "spine2",
    "left_ankle","right_ankle","spine3",
    "left_foot", "right_foot", "neck",
    "left_collar","right_collar","head",
    "left_shoulder","right_shoulder",
    "left_elbow","right_elbow",
    "left_wrist","right_wrist",
    "left_hand", "right_hand",
]
SMPL24_PARENTS  = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]
SMPL24_SKELETON = [[SMPL24_PARENTS[i], i] for i in range(1, 24)]

# ── MHR lower-body (v2) ───────────────────────────────────────────────────────

# Rest-pose local bone directions (fixed; precomputed from MHR template at zero pose).
# world_dir(t) = R_bone[t] @ LOCAL_DIR
L_FOOT_POSTERIOR_LOCAL = np.array([-0.2709380090236664,  0.9618300199508667, -0.0384100005030632])
R_FOOT_POSTERIOR_LOCAL = np.array([ 0.2709380090236664, -0.9618300199508667,  0.0384100005030632])
L_FOOT_INFERIOR_LOCAL  = np.array([ 0.9882242807,       -0.0992684317,       -0.1164411847      ])
R_FOOT_INFERIOR_LOCAL  = np.array([-0.9882242896,        0.0992684386,        0.1164411034      ])
L_BALL_DISTAL_LOCAL    = np.array([-0.9142919778823853, -0.3800260126590729, -0.1401830017566681])
R_BALL_DISTAL_LOCAL    = np.array([ 0.9142910242080688,  0.3800260126590729,  0.1401830017566681])

HEEL_TIP_DIST     = 0.055   # m — calcaneus center → posterior tip
HEEL_CONTACT_POST = 0.050   # m — posterior offset for contact point
HEEL_CONTACT_INF  = 0.110   # m — inferior offset for contact point
TOE_DIST          = 0.035   # m — MTP joint → toe tip

JOINT_NAMES_V2 = [
    "root",
    "l_upleg",  "l_lowleg",
    "l_foot",   "l_heel_tip",  "l_heel_contact",
    "l_talocrural", "l_subtalar", "l_transversetarsal", "l_ball", "l_toe_tip",
    "r_upleg",  "r_lowleg",
    "r_foot",   "r_heel_tip",  "r_heel_contact",
    "r_talocrural", "r_subtalar", "r_transversetarsal", "r_ball", "r_toe_tip",
]

JOINT_DESCRIPTIONS = {
    "root":               "Pelvis center (midpoint between hip joints)",
    "l_upleg":            "Left hip joint (ball of femur at pelvis)",
    "l_lowleg":           "Left knee joint (distal femur / proximal tibia)",
    "l_foot":             "Left calcaneus center (MHR subtalar attachment point)",
    "l_heel_tip":         "Left posterior calcaneal tip — back edge of heel (5.5 cm from l_foot via bone rotation)",
    "l_heel_contact":     "Left heel-strike contact point — inferior-posterior corner of calcaneus",
    "l_talocrural":       "Left ankle — tibio-talar hinge joint",
    "l_subtalar":         "Left subtalar joint (talus–calcaneus; inversion/eversion)",
    "l_transversetarsal": "Left midtarsal / Chopart joint",
    "l_ball":             "Left metatarsophalangeal (MTP) joint — ball of foot",
    "l_toe_tip":          "Left toe tip — 3.5 cm distal of l_ball via MTP bone rotation",
    "r_upleg":            "Right hip joint",
    "r_lowleg":           "Right knee joint",
    "r_foot":             "Right calcaneus center",
    "r_heel_tip":         "Right posterior calcaneal tip (see l_heel_tip)",
    "r_heel_contact":     "Right heel-strike contact point (see l_heel_contact)",
    "r_talocrural":       "Right tibio-talar ankle joint",
    "r_subtalar":         "Right subtalar joint",
    "r_transversetarsal": "Right midtarsal / Chopart joint",
    "r_ball":             "Right MTP joint — ball of foot",
    "r_toe_tip":          "Right toe tip (see l_toe_tip)",
}

MRIDULA_HIGHLIGHT = {
    "l_heel_contact": "L heel-strike contact point — primary gait event landmark",
    "l_heel_tip":     "L posterior heel tip — back edge of heel counter",
    "l_ball":         "L ball-of-foot — forefoot push-off contact",
    "l_toe_tip":      "L toe tip — distal foot boundary",
    "r_heel_contact": "R heel-strike contact point",
    "r_heel_tip":     "R posterior heel tip",
    "r_ball":         "R ball-of-foot",
    "r_toe_tip":      "R toe tip",
}

# idx:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
# jn: root lu  ll  lf lht lhc ltc lsb ltm lbl ltt ru  rl  rf rht rhc rtc rsb rtm rbl rtt
SKELETON_EDGES_V2 = [
    [0,  1],  [1,  2],  [2,  3],
    [3,  4],  [3,  5],
    [3,  6],  [6,  7],  [7,  8],  [8,  9],  [9, 10],
    [0, 11],  [11, 12], [12, 13],
    [13, 14], [13, 15],
    [13, 16], [16, 17], [17, 18], [18, 19], [19, 20],
]


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _norm(v):
    """Normalise last axis, (T,3) or (3,)."""
    n = np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-8)
    return v / n


def bone_tip(anchor, rot, *dir_dist_pairs):
    """anchor (T,3) + sum_i R[t]@local_dir_i * dist_i"""
    result = anchor.copy()
    for local_dir, dist in dir_dist_pairs:
        result = result + dist * np.einsum('tij,j->ti', rot, local_dir)
    return result


def heel_tip_fallback(lf, lb):
    fv = lf - lb; fv[:, 1] = 0.0
    return lf + HEEL_TIP_DIST * fv / np.linalg.norm(fv, axis=1, keepdims=True).clip(1e-6)


def heel_contact_fallback(lf, lb):
    fv = lf - lb; fv[:, 1] = 0.0
    post = fv / np.linalg.norm(fv, axis=1, keepdims=True).clip(1e-6)
    inf  = np.zeros_like(lf); inf[:, 1] = 1.0   # Y-DOWN = inferior
    return lf + HEEL_CONTACT_POST * post + HEEL_CONTACT_INF * inf


def toe_tip_fallback(tmt, lb):
    d = lb - tmt
    return lb + TOE_DIST * d / np.linalg.norm(d, axis=1, keepdims=True).clip(1e-6)


def project_pt(pt, cam_R, cam_t, focal):
    pc = cam_R @ pt + cam_t
    if pc[2] < 0.1:
        return None
    return [float(focal * pc[0] / pc[2] + W / 2),
            float(focal * pc[1] / pc[2] + H / 2)]


# ── Dataset v1 — SMPL-24 ─────────────────────────────────────────────────────

def export_smpl(athlete, rd):
    """Write sprint_smpl_dataset/data/{athlete}.json"""
    opt   = rd["joint_cam_human_optim"]
    J_wd  = np.array(opt["joints_world"], dtype=np.float64)   # (T,24,3) Y-DOWN
    cam_R = opt["cam_R"].astype(np.float64)                   # (T,3,3)
    cam_t = opt["cam_tvec"].astype(np.float64)                 # (T,3)
    focal = rd["camera_intrinsics_used"]["focal_length_px"].astype(np.float64)  # (T,)
    T     = J_wd.shape[0]

    sprint_dir = np.array(opt["sprint_dir"], dtype=np.float64).ravel()[:3]
    world_up   = np.array([0.0, -1.0, 0.0])   # physical up in Y-DOWN camera frame

    frames = []
    for t in range(T):
        J  = J_wd[t]           # (24,3)
        foc = float(focal[t])
        cR  = cam_R[t]; ct = cam_t[t]

        # Project all 24 joints to 2-D
        kpts_2d = []
        for j in range(24):
            pc = cR @ J[j] + ct
            if pc[2] > 0.1:
                kpts_2d.append([round(foc * pc[0] / pc[2] + W / 2, 1),
                                 round(foc * pc[1] / pc[2] + H / 2, 1)])
            else:
                kpts_2d.append([None, None])

        # Orientation vectors (Y-DOWN world frame — same as existing mridula_export)
        pelvis = J[0]; head = J[15]; l_sho = J[16]; r_sho = J[17]
        su = _norm((head - pelvis)[None])[0]                       # spine_up
        br = r_sho - l_sho
        br = _norm((br - np.dot(br, su) * su)[None])[0]           # body_right
        cf = np.cross(su, br)                                       # chest_fwd

        # Scalar angles
        sd_xz = np.array([sprint_dir[0], 0.0, sprint_dir[2]])
        cf_xz = np.array([cf[0], 0.0, cf[2]])
        sd_n  = sd_xz / np.linalg.norm(sd_xz).clip(1e-8)
        cf_n  = cf_xz / np.linalg.norm(cf_xz).clip(1e-8)
        facing_yaw_deg   = round(float(np.degrees(np.arccos(np.clip(np.dot(sd_n, cf_n), -1, 1)))), 2)
        spine_vertical_deg = round(float(np.degrees(np.arccos(np.clip(np.dot(su, world_up), -1, 1)))), 2)

        frames.append({
            "frame":              t,
            "kpts_2d":            kpts_2d,
            "spine_up":           [round(x, 4) for x in su.tolist()],
            "chest_fwd":          [round(x, 4) for x in cf.tolist()],
            "body_right":         [round(x, 4) for x in br.tolist()],
            "facing_yaw_deg":     facing_yaw_deg,
            "spine_vertical_deg": spine_vertical_deg,
        })

    out = {
        "athlete":    athlete,
        "fps":        30.0,
        "n_frames":   T,
        "image_size": [W, H],
        "world_up":   world_up.tolist(),
        "sprint_dir": [round(x, 4) for x in sprint_dir.tolist()],
        "joint_names": SMPL24_NAMES,
        "skeleton":    SMPL24_SKELETON,
        "frames":      frames,
    }
    os.makedirs(OUT_V1, exist_ok=True)
    path = os.path.join(OUT_V1, f"{athlete}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"    v1 → {path}  ({os.path.getsize(path)/1e6:.1f} MB)")


# ── Dataset v2 — MHR lower-body ──────────────────────────────────────────────

def export_lowerbody(athlete, mk, rd):
    """Write sprint_lowerbody_dataset/data/{athlete}.json"""
    jnames = mk["joint_names"]
    J_all  = mk["all_joint_positions_world_m"]     # (T,J,3) Y-DOWN, meters
    T      = J_all.shape[0]

    cam_R = rd["joint_cam_human_optim"]["cam_R"].astype(np.float64)
    cam_t = rd["joint_cam_human_optim"]["cam_tvec"].astype(np.float64)
    focal = rd["camera_intrinsics_used"]["focal_length_px"].astype(np.float64)

    fi_map = {n: jnames.index(n) for n in jnames}
    y_floor = float(np.max([J_all[:, fi_map[n], 1]
                             for n in ["l_foot", "r_foot", "l_ball", "r_ball"]]))

    def yup(p):
        q = p.copy(); q[..., 1] = y_floor - q[..., 1]; return q

    SKEL_KWS = ["root","l_upleg","l_lowleg","l_foot","l_talocrural","l_subtalar",
                "l_transversetarsal","l_ball","r_upleg","r_lowleg","r_foot",
                "r_talocrural","r_subtalar","r_transversetarsal","r_ball"]
    jidx = {n: fi_map[n] for n in SKEL_KWS if n in fi_map}

    lf = J_all[:, jidx["l_foot"], :]; rf = J_all[:, jidx["r_foot"], :]
    lb = J_all[:, jidx["l_ball"], :]; rb = J_all[:, jidx["r_ball"], :]
    ltmt = J_all[:, jidx["l_transversetarsal"], :]
    rtmt = J_all[:, jidx["r_transversetarsal"], :]

    has_bone = "l_foot_world_rot" in mk
    mode = "bone-rotation" if has_bone else "geometric-fallback"

    if has_bone:
        l_ht = bone_tip(lf, mk["l_foot_world_rot"], (L_FOOT_POSTERIOR_LOCAL, HEEL_TIP_DIST))
        r_ht = bone_tip(rf, mk["r_foot_world_rot"], (R_FOOT_POSTERIOR_LOCAL, HEEL_TIP_DIST))
        l_hc = bone_tip(lf, mk["l_foot_world_rot"], (L_FOOT_POSTERIOR_LOCAL, HEEL_CONTACT_POST),
                                                     (L_FOOT_INFERIOR_LOCAL,  HEEL_CONTACT_INF))
        r_hc = bone_tip(rf, mk["r_foot_world_rot"], (R_FOOT_POSTERIOR_LOCAL, HEEL_CONTACT_POST),
                                                     (R_FOOT_INFERIOR_LOCAL,  HEEL_CONTACT_INF))
        l_tt = bone_tip(lb, mk["l_ball_world_rot"], (L_BALL_DISTAL_LOCAL, TOE_DIST))
        r_tt = bone_tip(rb, mk["r_ball_world_rot"], (R_BALL_DISTAL_LOCAL, TOE_DIST))
    else:
        l_ht = heel_tip_fallback(lf, lb);    r_ht = heel_tip_fallback(rf, rb)
        l_hc = heel_contact_fallback(lf, lb); r_hc = heel_contact_fallback(rf, rb)
        l_tt = toe_tip_fallback(ltmt, lb);   r_tt = toe_tip_fallback(rtmt, rb)

    mid = T // 2
    for name, tip, anchor in [("l_heel_tip", l_ht, lf),
                               ("l_heel_contact", l_hc, lf),
                               ("l_toe_tip", l_tt, lb)]:
        ty = yup(tip[mid]); ay = yup(anchor[mid])
        print(f"    {name:<20s}  3D={np.linalg.norm(ty-ay)*100:.1f}cm"
              f"  dY={( ty[1]-ay[1])*100:+.1f}cm")

    extra = {"l_heel_tip": l_ht, "r_heel_tip": r_ht,
             "l_heel_contact": l_hc, "r_heel_contact": r_hc,
             "l_toe_tip": l_tt, "r_toe_tip": r_tt}

    joints_out = {}
    for jname in JOINT_NAMES_V2:
        pos = extra[jname] if jname in extra else \
              J_all[:, jidx[jname], :] if jname in jidx else None
        if pos is None:
            continue
        pixels = [project_pt(pos[t], cam_R[t], cam_t[t], float(focal[t])) or [None, None]
                  for t in range(T)]
        joints_out[jname] = {"world_m": yup(pos).tolist(), "pixel_uv": pixels}

    out = {
        "athlete":            athlete,
        "n_frames":           T,
        "fps":                30.0,
        "tip_computation":    mode,
        "heel_tip_dist_m":    HEEL_TIP_DIST,
        "heel_contact_post_m":HEEL_CONTACT_POST,
        "heel_contact_inf_m": HEEL_CONTACT_INF,
        "toe_dist_m":         TOE_DIST,
        "coordinate_system": {
            "world_m":   "Y-up, meters. Y=0 is floor. Athlete runs roughly along +X.",
            "pixel_uv":  "Origin=top-left. u=right, v=down.",
        },
        "joint_names":        JOINT_NAMES_V2,
        "joint_descriptions": JOINT_DESCRIPTIONS,
        "mridula_highlight":  MRIDULA_HIGHLIGHT,
        "skeleton_edges":     SKELETON_EDGES_V2,
        "camera": {
            "image_width":       W,
            "image_height":      H,
            "focal_length_px":   focal.tolist(),
            "principal_point_px":[W / 2, H / 2],
            "cam_R":             cam_R.tolist(),
            "cam_t":             cam_t.tolist(),
            "y_floor_ydown":     y_floor,
            "projection_note": (
                "cam_R/cam_t operate on Y-DOWN coords. "
                "To project a Y-UP point p: p_yd[1]=y_floor_ydown-p[1]; "
                "pc=cam_R[t]@p_yd+cam_t[t]; u=focal[t]*pc[0]/pc[2]+W/2; v=focal[t]*pc[1]/pc[2]+H/2."
            ),
        },
        "joints": joints_out,
    }
    os.makedirs(OUT_V2, exist_ok=True)
    path = os.path.join(OUT_V2, f"{athlete}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"    v2 → {path}  ({os.path.getsize(path)/1e6:.1f} MB, {len(joints_out)} joints)")


# ── Main ──────────────────────────────────────────────────────────────────────

def export_athlete(athlete):
    pkl_path = os.path.join(ROLLOUT_DIR, athlete, "mhr_kpts.pkl")
    rd_path  = os.path.join(ROLLOUT_DIR, athlete, "rollout.pkl")

    if not os.path.exists(rd_path):
        print(f"  SKIP {athlete} — rollout.pkl not found"); return
    has_mhr = os.path.exists(pkl_path)
    if not has_mhr:
        print(f"  WARN {athlete} — mhr_kpts.pkl not found; v2 skipped")

    with open(rd_path, "rb") as f:
        rd = pickle.load(f)

    T = np.array(rd["joint_cam_human_optim"]["joints_world"]).shape[0]
    mk = None
    if has_mhr:
        with open(pkl_path, "rb") as f:
            mk = pickle.load(f)

    mode = "bone-rotation" if (mk is not None and "l_foot_world_rot" in mk) else "geometric-fallback"
    print(f"  {athlete}: T={T}  mhr={'yes' if has_mhr else 'no'}  mode={mode}")

    export_smpl(athlete, rd)
    if mk is not None:
        export_lowerbody(athlete, mk, rd)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--athlete", default="", choices=[""] + ATHLETES,
                    help="Single athlete (default: all)")
    args = ap.parse_args()
    athletes = [args.athlete] if args.athlete else ATHLETES
    print(f"Exporting: {athletes}")
    for ath in athletes:
        export_athlete(ath)
    print("Done.")


if __name__ == "__main__":
    main()
