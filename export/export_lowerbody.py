#!/usr/bin/env python3
"""
export_lowerbody.py — Export MHR lower-body keypoints to per-athlete JSON files.

Reads:
  ../../outputs_with_moge2/rollout_results/{Athlete}/mhr_kpts.pkl
  ../../outputs_with_moge2/rollout_results/{Athlete}/rollout.pkl

Writes:
  ../../sprint_lowerbody_dataset/data/{Athlete}.json

Joint set (21 joints)
─────────────────────
  root
  l_upleg, l_lowleg
  l_foot, l_heel_tip*, l_heel_contact*,
  l_talocrural, l_subtalar, l_transversetarsal, l_ball, l_toe_tip*
  r_upleg, r_lowleg
  r_foot, r_heel_tip*, r_heel_contact*,
  r_talocrural, r_subtalar, r_transversetarsal, r_ball, r_toe_tip*

*Derived joints from bone rotation matrices (smpl_to_mhr_kpts.py).
 Falls back to geometric approximation if pkl missing bone rots.

Derived joint definitions
─────────────────────────
  l_heel_tip      = l_foot + R_lfoot @ POSTERIOR_LOCAL * 5.5 cm
                    Posterior calcaneal tip — back edge of heel/shoe counter (upper).

  l_heel_contact  = l_foot + R_lfoot @ POSTERIOR_LOCAL * 3.0 cm
                            + R_lfoot @ INFERIOR_LOCAL  * 2.5 cm
                    Inferior-posterior calcaneal tuberosity — the actual
                    heel-strike contact point at the bottom-back corner.

  l_toe_tip       = l_ball + R_lball @ DISTAL_LOCAL * 3.5 cm
                    Estimated distal phalanx tip beyond the MTP joint.

Rest-pose local directions (computed from MHR skeleton, model_params=zeros, Y-UP world)
─────────────────────────────────────────────────────────────────────────────────────────
  L_FOOT_POSTERIOR_LOCAL (-0.2709,  0.9618, -0.0384)  → posterior of calcaneus
  L_FOOT_INFERIOR_LOCAL  ( 0.9882, -0.0993, -0.1164)  → plantar/inferior of calcaneus
  L_BALL_DISTAL_LOCAL    (-0.9143, -0.3800, -0.1402)  → distal beyond MTP

Run:
  conda activate sam_3d_body
  python export_lowerbody.py            # all 7 athletes
  python export_lowerbody.py --athlete Original
"""
import argparse, json, os, pickle
import numpy as np

HERE        = os.path.dirname(os.path.abspath(__file__))
ROLLOUT_DIR = os.path.join(HERE, "..", "..", "outputs_with_moge2", "rollout_results")
OUT_DIR     = os.path.join(HERE, "..", "..", "sprint_lowerbody_dataset", "data")
ATHLETES    = ["Bishop", "Colin_Brazzell", "Goree", "Jackson", "Original", "Poteat", "Walton"]

# ── Rest-pose local directions ────────────────────────────────────────────────
# Each vector is in the respective bone's LOCAL frame (constant across all poses).
# Apply via: world_dir = R_bone[t] @ LOCAL_DIR
L_FOOT_POSTERIOR_LOCAL = np.array([-0.2709380090236664,  0.9618300199508667, -0.0384100005030632])
R_FOOT_POSTERIOR_LOCAL = np.array([ 0.2709380090236664, -0.9618300199508667,  0.0384100005030632])
L_FOOT_INFERIOR_LOCAL  = np.array([ 0.9882242807,       -0.0992684317,       -0.1164411847      ])
R_FOOT_INFERIOR_LOCAL  = np.array([-0.9882242896,        0.0992684386,        0.1164411034      ])
L_BALL_DISTAL_LOCAL    = np.array([-0.9142919778823853, -0.3800260126590729, -0.1401830017566681])
R_BALL_DISTAL_LOCAL    = np.array([ 0.9142910242080688,  0.3800260126590729,  0.1401830017566681])

# ── Extension distances ───────────────────────────────────────────────────────
HEEL_TIP_DIST    = 0.055   # m — calcaneus center → posterior tip (back of heel counter)
HEEL_CONTACT_POST = 0.050  # m — posterior component of contact point
HEEL_CONTACT_INF  = 0.110  # m — inferior component of contact point
TOE_DIST          = 0.035  # m — MTP → toe tip

# ── Joint set (21 joints) ─────────────────────────────────────────────────────
JOINT_NAMES = [
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
    "l_foot":             "Left calcaneus center (MHR skeleton attachment point / subtalar joint)",
    "l_heel_tip":         ("Left POSTERIOR calcaneal tip — the very back of the heel/shoe counter "
                           "(upper edge). 5.5 cm extension in l_foot bone's posterior direction."),
    "l_heel_contact":     ("Left INFERIOR-POSTERIOR calcaneal tuberosity — the heel-strike CONTACT "
                           "POINT at the bottom-back corner of the calcaneus. Computed as: "
                           "3.0 cm posterior + 2.5 cm inferior from calcaneus center, "
                           "following the bone's orientation."),
    "l_talocrural":       "Left ankle — tibio-talar joint (true ankle hinge)",
    "l_subtalar":         "Left subtalar joint (talus-calcaneus; inversion/eversion axis)",
    "l_transversetarsal": "Left midtarsal / Chopart joint (talonavicular + calcaneocuboid)",
    "l_ball":             ("Left ball-of-foot — metatarsophalangeal (MTP) joint / toe knuckle. "
                           "Most distal foot joint in MHR skeleton."),
    "l_toe_tip":          "Left toe tip — 3.5 cm extension beyond l_ball along distal MTP axis.",
    "r_upleg":            "Right hip joint",
    "r_lowleg":           "Right knee joint",
    "r_foot":             "Right calcaneus center",
    "r_heel_tip":         "Right POSTERIOR calcaneal tip (see l_heel_tip).",
    "r_heel_contact":     "Right INFERIOR-POSTERIOR calcaneal tuberosity — heel-strike contact point (see l_heel_contact).",
    "r_talocrural":       "Right ankle (tibio-talar)",
    "r_subtalar":         "Right subtalar joint",
    "r_transversetarsal": "Right midtarsal / Chopart joint",
    "r_ball":             "Right ball-of-foot (MTP / toe knuckle)",
    "r_toe_tip":          "Right toe tip — 3.5 cm extension beyond r_ball.",
}

# idx:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
# jn: root lu  ll  lf lht lhc ltc lsb ltm lbl ltt ru  rl  rf rht rhc rtc rsb rtm rbl rtt
SKELETON_EDGES = [
    [0,  1],  [1,  2],  [2,  3],          # root → l_upleg → l_lowleg → l_foot
    [3,  4],                               # l_foot → l_heel_tip    (posterior tip branch)
    [3,  5],                               # l_foot → l_heel_contact (contact branch)
    [3,  6],  [6,  7],  [7,  8],  [8,  9], # l_foot → ankle → subtalar → midtarsal → l_ball
    [9, 10],                               # l_ball → l_toe_tip      (toe tip branch)
    [0, 11],  [11, 12], [12, 13],          # root → r_upleg → r_lowleg → r_foot
    [13, 14],                              # r_foot → r_heel_tip    (posterior tip branch)
    [13, 15],                              # r_foot → r_heel_contact (contact branch)
    [13, 16], [16, 17], [17, 18], [18, 19], # r_foot → ankle chain → r_ball
    [19, 20],                              # r_ball → r_toe_tip      (toe tip branch)
]

MRIDULA_HIGHLIGHT = {
    "l_heel_contact": "L inferior-posterior heel — primary heel-strike contact point",
    "l_heel_tip":     "L posterior heel tip — back edge of heel counter",
    "l_ball":         "L ball-of-foot — forefoot push-off contact",
    "l_toe_tip":      "L toe tip — distal foot boundary",
    "r_heel_contact": "R inferior-posterior heel — primary heel-strike contact point",
    "r_heel_tip":     "R posterior heel tip — back edge of heel counter",
    "r_ball":         "R ball-of-foot — forefoot push-off contact",
    "r_toe_tip":      "R toe tip — distal foot boundary",
}


# ── Geometry ──────────────────────────────────────────────────────────────────

def bone_tip(anchor, rot, *dir_dist_pairs):
    """anchor (T,3) + sum of R@local_dir*dist for each (local_dir, dist) pair."""
    result = anchor.copy()
    for local_dir, dist in dir_dist_pairs:
        world_dir = np.einsum('tij,j->ti', rot, local_dir)
        result = result + dist * world_dir
    return result


def heel_tip_fallback(lf, lb):
    fv = lf - lb; fv[:, 1] = 0.0
    n  = np.linalg.norm(fv, axis=1, keepdims=True).clip(1e-6)
    return lf + HEEL_TIP_DIST * fv / n


def heel_contact_fallback(lf, lb):
    # Posterior: horizontal foot direction, 3cm back
    fv = lf - lb; fv[:, 1] = 0.0
    n  = np.linalg.norm(fv, axis=1, keepdims=True).clip(1e-6)
    post = fv / n
    # Inferior: world down (Y-DOWN = +Y axis)
    inf = np.zeros_like(lf); inf[:, 1] = 1.0
    return lf + HEEL_CONTACT_POST * post + HEEL_CONTACT_INF * inf


def toe_tip_fallback(tmt, lb):
    d = lb - tmt
    n = np.linalg.norm(d, axis=1, keepdims=True).clip(1e-6)
    return lb + TOE_DIST * d / n


def project(pt_ydown, cam_R, cam_t, focal, W, H):
    pc = cam_R @ pt_ydown + cam_t
    if pc[2] < 0.1: return None
    return [float(focal * pc[0] / pc[2] + W / 2),
            float(focal * pc[1] / pc[2] + H / 2)]


# ── Per-athlete export ────────────────────────────────────────────────────────

def export_athlete(athlete):
    pkl_path = os.path.join(ROLLOUT_DIR, athlete, "mhr_kpts.pkl")
    rd_path  = os.path.join(ROLLOUT_DIR, athlete, "rollout.pkl")
    out_path = os.path.join(OUT_DIR, f"{athlete}.json")

    if not os.path.exists(pkl_path) or not os.path.exists(rd_path):
        print(f"  SKIP {athlete} — pkl not found"); return

    with open(pkl_path, "rb") as f:  mk = pickle.load(f)
    with open(rd_path,  "rb") as f:  rd = pickle.load(f)

    jnames = mk["joint_names"]
    J_all  = mk["all_joint_positions_world_m"]   # (T, J, 3) Y-DOWN, meters
    T      = J_all.shape[0]

    cam_R  = rd["joint_cam_human_optim"]["cam_R"].astype(np.float64)
    cam_t  = rd["joint_cam_human_optim"]["cam_tvec"].astype(np.float64)
    focal  = rd["camera_intrinsics_used"]["focal_length_px"].astype(np.float64)
    W, H   = 1920, 1080

    has_bone = "l_foot_world_rot" in mk
    mode = "bone-rotation" if has_bone else "geometric-fallback"
    print(f"  {athlete}: T={T}  mode={mode}")

    # Floor
    fi_map   = {n: jnames.index(n) for n in jnames}
    y_floor  = float(np.max([J_all[:, fi_map[n], 1]
                              for n in ["l_foot","r_foot","l_ball","r_ball"]]))

    def yup(p): q = p.copy(); q[..., 1] = y_floor - q[..., 1]; return q

    SKEL = ["root","l_upleg","l_lowleg","l_foot","l_talocrural","l_subtalar",
            "l_transversetarsal","l_ball","r_upleg","r_lowleg","r_foot",
            "r_talocrural","r_subtalar","r_transversetarsal","r_ball"]
    jidx = {n: fi_map[n] for n in SKEL if n in fi_map}

    lf  = J_all[:, jidx["l_foot"], :]
    rf  = J_all[:, jidx["r_foot"], :]
    lb  = J_all[:, jidx["l_ball"], :]
    rb  = J_all[:, jidx["r_ball"], :]
    ltmt = J_all[:, jidx["l_transversetarsal"], :]
    rtmt = J_all[:, jidx["r_transversetarsal"], :]

    if has_bone:
        lf_rot = mk["l_foot_world_rot"]
        rf_rot = mk["r_foot_world_rot"]
        lb_rot = mk["l_ball_world_rot"]
        rb_rot = mk["r_ball_world_rot"]
        l_ht  = bone_tip(lf, lf_rot, (L_FOOT_POSTERIOR_LOCAL, HEEL_TIP_DIST))
        r_ht  = bone_tip(rf, rf_rot, (R_FOOT_POSTERIOR_LOCAL, HEEL_TIP_DIST))
        l_hc  = bone_tip(lf, lf_rot, (L_FOOT_POSTERIOR_LOCAL, HEEL_CONTACT_POST),
                                      (L_FOOT_INFERIOR_LOCAL,  HEEL_CONTACT_INF))
        r_hc  = bone_tip(rf, rf_rot, (R_FOOT_POSTERIOR_LOCAL, HEEL_CONTACT_POST),
                                      (R_FOOT_INFERIOR_LOCAL,  HEEL_CONTACT_INF))
        l_tt  = bone_tip(lb, lb_rot, (L_BALL_DISTAL_LOCAL, TOE_DIST))
        r_tt  = bone_tip(rb, rb_rot, (R_BALL_DISTAL_LOCAL, TOE_DIST))
    else:
        l_ht  = heel_tip_fallback(lf, lb)
        r_ht  = heel_tip_fallback(rf, rb)
        l_hc  = heel_contact_fallback(lf, lb)
        r_hc  = heel_contact_fallback(rf, rb)
        l_tt  = toe_tip_fallback(ltmt, lb)
        r_tt  = toe_tip_fallback(rtmt, rb)

    mid = T // 2
    for name, tip, anchor in [("l_heel_tip", l_ht, lf), ("l_heel_contact", l_hc, lf),
                               ("l_toe_tip", l_tt, lb)]:
        ty = yup(tip[mid]); ay = yup(anchor[mid])
        d3 = np.linalg.norm(ty - ay)
        dy = ty[1] - ay[1]
        print(f"    {name:<20s}  3D={d3*100:.1f}cm  dY={dy*100:+.1f}cm from anchor")

    extra = {"l_heel_tip": l_ht, "r_heel_tip": r_ht,
             "l_heel_contact": l_hc, "r_heel_contact": r_hc,
             "l_toe_tip": l_tt, "r_toe_tip": r_tt}

    joints_out = {}
    for jname in JOINT_NAMES:
        pos = extra[jname] if jname in extra else \
              J_all[:, jidx[jname], :] if jname in jidx else None
        if pos is None: continue
        pixels = [project(pos[t], cam_R[t], cam_t[t], float(focal[t]), W, H) or [None,None]
                  for t in range(T)]
        joints_out[jname] = {"world_m": yup(pos).tolist(), "pixel_uv": pixels}

    camera_out = {
        "image_width": W, "image_height": H,
        "focal_length_px": focal.tolist(),
        "principal_point_px": [W/2, H/2],
        "cam_R": cam_R.tolist(), "cam_t": cam_t.tolist(),
        "y_floor_ydown": y_floor,
        "projection_note": (
            "cam_R/cam_t operate on Y-DOWN coords. "
            "To project Y-UP point: pt_ydown[1]=y_floor_ydown-pt_yup[1]; "
            "pt_cam=cam_R[t]@pt_ydown+cam_t[t]; "
            "u=focal[t]*pt_cam[0]/pt_cam[2]+W/2; v=focal[t]*pt_cam[1]/pt_cam[2]+H/2."
        ),
    }

    out = {
        "athlete": athlete, "n_frames": T, "fps": 30.0,
        "tip_computation": mode,
        "heel_tip_dist_m": HEEL_TIP_DIST,
        "heel_contact_post_m": HEEL_CONTACT_POST,
        "heel_contact_inf_m": HEEL_CONTACT_INF,
        "toe_dist_m": TOE_DIST,
        "coordinate_system": {
            "world_m": "Y-up, meters. Y=0 is floor. Athlete runs roughly along +X.",
            "pixel_uv": "Origin=top-left. u=right, v=down.",
        },
        "joint_names": JOINT_NAMES,
        "joint_descriptions": JOINT_DESCRIPTIONS,
        "mridula_highlight": MRIDULA_HIGHLIGHT,
        "skeleton_edges": SKELETON_EDGES,
        "camera": camera_out,
        "joints": joints_out,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(out_path, "w") as f: json.dump(out, f, indent=2)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"    → {out_path}  ({size_mb:.1f} MB, {len(joints_out)} joints)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--athlete", default="", choices=[""] + ATHLETES)
    args = ap.parse_args()
    athletes = [args.athlete] if args.athlete else ATHLETES
    print(f"Exporting: {athletes}")
    for ath in athletes: export_athlete(ath)
    print("Done.")

if __name__ == "__main__":
    main()
