#!/usr/bin/env python3
"""
tune_heel.py — Live hyperparameter tuner for heel_contact / heel_tip / toe_tip.

Loads pkl data directly; recomputes + re-projects derived joints instantly on each
keypress (no subprocess). Press 'r' to write data/{Athlete}.json. Press 'w' to
save current params back into export_lowerbody.py.

Controls
────────
  z / x      : HEEL_TIP_DIST      − / + 0.005 m
  c / v      : HEEL_CONTACT_POST  − / + 0.005 m
  b / n      : HEEL_CONTACT_INF   − / + 0.005 m
  m / ,      : TOE_DIST           − / + 0.005 m
  ← / →      : prev / next frame  (also A / D)
  + / -      : ± 10 frames
  0 – 9      : jump to 0 % – 90 %
  e          : export overlay video → overlays/{Athlete}_tuned_*.mp4
  r          : re-export JSON → data/{Athlete}.json
  w          : write current params into export_lowerbody.py
  l          : toggle labels
  q / Esc    : quit

Usage:
  conda activate sam_3d_body
  cd /home/nan/Desktop/NRMFOptim/Athlete_Mesh_Analysis_Codes/export
  python tune_heel.py --athlete Goree
  python tune_heel.py --athlete Goree --joints heel_contact,heel_tip
"""
import argparse, json, os, pickle, re, sys
import cv2
import numpy as np

HERE        = os.path.dirname(os.path.abspath(__file__))
ROLLOUT_DIR = os.path.join(HERE, "..", "..", "outputs_with_moge2", "rollout_results")
DATA_DIR    = os.path.join(HERE, "..", "..", "sprint_lowerbody_dataset", "data")
VID_DIR     = os.environ.get("DATASET_DIR", "")
EXPORT_PY   = os.path.join(HERE, "export_lowerbody.py")
ATHLETES    = ["Bishop", "Colin_Brazzell", "Goree", "Jackson", "Original", "Poteat", "Walton"]

# ── Fixed rest-pose local directions (from MHR rest pose, never change) ───────
L_FOOT_POSTERIOR_LOCAL = np.array([-0.2709380090236664,  0.9618300199508667, -0.0384100005030632])
R_FOOT_POSTERIOR_LOCAL = np.array([ 0.2709380090236664, -0.9618300199508667,  0.0384100005030632])
L_FOOT_INFERIOR_LOCAL  = np.array([ 0.9882242807,       -0.0992684317,       -0.1164411847      ])
R_FOOT_INFERIOR_LOCAL  = np.array([-0.9882242896,        0.0992684386,        0.1164411034      ])
L_BALL_DISTAL_LOCAL    = np.array([-0.9142919778823853, -0.3800260126590729, -0.1401830017566681])
R_BALL_DISTAL_LOCAL    = np.array([ 0.9142910242080688,  0.3800260126590729,  0.1401830017566681])

# ── Colors BGR ────────────────────────────────────────────────────────────────
C_BLACK        = (  0,   0,   0)
C_WHITE        = (240, 240, 240)
C_GREEN        = ( 50, 220,  60)
C_HEEL_RING    = (  0, 200, 255)
C_BALL_RING    = (  0, 255, 200)
C_HEEL_TIP     = (  0,  80, 255)
C_HEEL_CONTACT = (  0, 165, 255)
C_TOE_TIP      = (130,  50, 255)
C_CHANGED      = ( 50, 255, 255)  # bright cyan — param just changed
FONT           = cv2.FONT_HERSHEY_SIMPLEX

STEP = 0.005   # m per keypress


# ── Geometry helpers ──────────────────────────────────────────────────────────

def bone_tip(anchor, rot, *dir_dist_pairs):
    result = anchor.copy()
    for local_dir, dist in dir_dist_pairs:
        world_dir = np.einsum('tij,j->ti', rot, local_dir)
        result = result + dist * world_dir
    return result


def heel_tip_fallback(lf, lb, dist):
    fv = lf - lb; fv[:, 1] = 0.0
    n  = np.linalg.norm(fv, axis=1, keepdims=True).clip(1e-6)
    return lf + dist * fv / n


def heel_contact_fallback(lf, lb, post, inf):
    fv = lf - lb; fv[:, 1] = 0.0
    n  = np.linalg.norm(fv, axis=1, keepdims=True).clip(1e-6)
    fv = fv / n
    down = np.zeros_like(lf); down[:, 1] = 1.0  # +Y = down in Y-DOWN
    return lf + post * fv + inf * down


def toe_tip_fallback(tmt, lb, dist):
    d = lb - tmt
    n = np.linalg.norm(d, axis=1, keepdims=True).clip(1e-6)
    return lb + dist * d / n


def project_pt(pt_ydown, cam_R, cam_t, focal, W, H):
    pc = cam_R @ pt_ydown + cam_t
    if pc[2] < 0.1: return None
    return (int(round(focal * pc[0] / pc[2] + W / 2)),
            int(round(focal * pc[1] / pc[2] + H / 2)))


def in_frame(px, W, H, margin=150):
    return px is not None and -margin < px[0] < W+margin and -margin < px[1] < H+margin


def txt(img, s, org, scale=0.36, color=C_WHITE, thick=1):
    cv2.putText(img, s, org, FONT, scale, C_BLACK, thick+2, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color,   thick,   cv2.LINE_AA)


# ── Recompute all derived joints for current params ───────────────────────────

def recompute(data, params):
    lf, rf, lb, rb, ltmt, rtmt = (data[k] for k in
        ["lf","rf","lb","rb","ltmt","rtmt"])
    has_bone = data["has_bone"]
    ht, hcp, hci, td = params["HEEL_TIP_DIST"], params["HEEL_CONTACT_POST"], \
                        params["HEEL_CONTACT_INF"], params["TOE_DIST"]

    if has_bone:
        lf_rot, rf_rot = data["lf_rot"], data["rf_rot"]
        lb_rot, rb_rot = data["lb_rot"], data["rb_rot"]
        l_ht = bone_tip(lf, lf_rot, (L_FOOT_POSTERIOR_LOCAL, ht))
        r_ht = bone_tip(rf, rf_rot, (R_FOOT_POSTERIOR_LOCAL, ht))
        l_hc = bone_tip(lf, lf_rot, (L_FOOT_POSTERIOR_LOCAL, hcp),
                                     (L_FOOT_INFERIOR_LOCAL,  hci))
        r_hc = bone_tip(rf, rf_rot, (R_FOOT_POSTERIOR_LOCAL, hcp),
                                     (R_FOOT_INFERIOR_LOCAL,  hci))
        l_tt = bone_tip(lb, lb_rot, (L_BALL_DISTAL_LOCAL, td))
        r_tt = bone_tip(rb, rb_rot, (R_BALL_DISTAL_LOCAL, td))
    else:
        l_ht = heel_tip_fallback(lf, lb, ht)
        r_ht = heel_tip_fallback(rf, rb, ht)
        l_hc = heel_contact_fallback(lf, lb, hcp, hci)
        r_hc = heel_contact_fallback(rf, rb, hcp, hci)
        l_tt = toe_tip_fallback(ltmt, lb, td)
        r_tt = toe_tip_fallback(rtmt, rb, td)

    return {"l_heel_tip": l_ht, "r_heel_tip": r_ht,
            "l_heel_contact": l_hc, "r_heel_contact": r_hc,
            "l_toe_tip": l_tt, "r_toe_tip": r_tt}


def get_pixels(derived, data, fi):
    cam_R = data["cam_R"][fi]
    cam_t = data["cam_t"][fi]
    focal = float(data["focal"][fi])
    W, H  = data["W"], data["H"]
    return {k: project_pt(v[fi], cam_R, cam_t, focal, W, H)
            for k, v in derived.items()}


def get_anchor_pixels(data, fi):
    cam_R = data["cam_R"][fi]
    cam_t = data["cam_t"][fi]
    focal = float(data["focal"][fi])
    W, H  = data["W"], data["H"]
    out = {}
    for k in ["lf","rf","lb","rb"]:
        out[k] = project_pt(data[k][fi], cam_R, cam_t, focal, W, H)
    return out


# ── Draw frame ────────────────────────────────────────────────────────────────

def draw(img, fi, derived_px, anchor_px, params, show_labels,
         show_set, W, H, athlete, T, mode, last_changed):

    # skeleton lines: foot→heel_tip, foot→heel_contact, ball→toe_tip
    pairs = [("lf","l_heel_tip"), ("lf","l_heel_contact"), ("lb","l_toe_tip"),
             ("rf","r_heel_tip"), ("rf","r_heel_contact"), ("rb","r_toe_tip")]
    amap  = {"lf": "l_foot", "rf": "r_foot", "lb": "l_ball", "rb": "r_ball"}
    for ak, jk in pairs:
        if jk not in show_set: continue
        pxa = anchor_px.get(ak); pxb = derived_px.get(jk)
        if in_frame(pxa,W,H) and in_frame(pxb,W,H):
            cv2.line(img, pxa, pxb, C_WHITE, 1, cv2.LINE_AA)

    # anchor dots
    for ak, col, ring in [("lf", C_GREEN, C_HEEL_RING), ("rf", C_GREEN, C_HEEL_RING),
                           ("lb", C_GREEN, C_BALL_RING), ("rb", C_GREEN, C_BALL_RING)]:
        aname = amap[ak]
        if aname not in show_set: continue
        px = anchor_px.get(ak)
        if not in_frame(px,W,H): continue
        cv2.circle(img, px, 1, C_BLACK, -1, cv2.LINE_AA)
        cv2.circle(img, px, 1, col,      1, cv2.LINE_AA)
        cv2.circle(img, px, 3, ring,     1, cv2.LINE_AA)
        if show_labels:
            txt(img, aname.replace("l_","L:").replace("r_","R:"),
                (px[0]+5, px[1]+4), 0.26, col)

    # derived joint dots
    for jname, px in derived_px.items():
        if jname not in show_set: continue
        if not in_frame(px,W,H): continue

        if "heel_tip" in jname:
            cv2.circle(img, px, 2, C_BLACK,   -1, cv2.LINE_AA)
            cv2.circle(img, px, 2, C_HEEL_TIP, 1, cv2.LINE_AA)
            cv2.circle(img, px, 4, C_HEEL_TIP, 1, cv2.LINE_AA)
            if show_labels:
                txt(img, jname.replace("l_","L:").replace("r_","R:"),
                    (px[0]+6, px[1]+4), 0.26, C_HEEL_TIP)

        elif "heel_contact" in jname:
            cv2.circle(img, px, 2, C_BLACK,       -1, cv2.LINE_AA)
            cv2.circle(img, px, 2, C_HEEL_CONTACT, 2, cv2.LINE_AA)
            cv2.circle(img, px, 4, C_HEEL_CONTACT, 1, cv2.LINE_AA)
            cv2.circle(img, px, 7, C_HEEL_CONTACT, 1, cv2.LINE_AA)
            if show_labels:
                txt(img, jname.replace("l_","L:").replace("r_","R:"),
                    (px[0]+9, px[1]+4), 0.26, C_HEEL_CONTACT)

        elif "toe_tip" in jname:
            cv2.circle(img, px, 2, C_BLACK,  -1, cv2.LINE_AA)
            cv2.circle(img, px, 2, C_TOE_TIP, 1, cv2.LINE_AA)
            cv2.circle(img, px, 4, C_TOE_TIP, 1, cv2.LINE_AA)
            if show_labels:
                txt(img, jname.replace("l_","L:").replace("r_","R:"),
                    (px[0]+6, px[1]+4), 0.26, C_TOE_TIP)

    # HUD — param panel (left side)
    ht  = params["HEEL_TIP_DIST"]
    hcp = params["HEEL_CONTACT_POST"]
    hci = params["HEEL_CONTACT_INF"]
    td  = params["TOE_DIST"]

    def pcol(key): return C_CHANGED if last_changed == key else C_WHITE

    hud = [
        (f"  {athlete}   frame {fi}/{T-1}   mode={mode}", C_WHITE),
        (f"  z/x  HEEL_TIP_DIST     {ht*100:5.1f} cm  (-/+)", pcol("HEEL_TIP_DIST")),
        (f"  c/v  HEEL_CONTACT_POST {hcp*100:5.1f} cm  (-/+)", pcol("HEEL_CONTACT_POST")),
        (f"  b/n  HEEL_CONTACT_INF  {hci*100:5.1f} cm  (-/+)", pcol("HEEL_CONTACT_INF")),
        (f"  m/,  TOE_DIST          {td*100:5.1f} cm  (-/+)", pcol("TOE_DIST")),
        (f"  e=export-video  r=export-json  w=save-py  l=labels  q=quit", (130,130,130)),
    ]
    for k, (line, col) in enumerate(hud):
        txt(img, line, (8, 22 + k*20), 0.40, col)

    # legend bottom-right
    H_, W_ = img.shape[:2]
    legend = [
        ("RED double ring    = heel_tip",              C_HEEL_TIP),
        ("ORANGE triple ring = heel_contact (CONTACT)", C_HEEL_CONTACT),
        ("MAG double ring    = toe_tip",               C_TOE_TIP),
        ("gold/cyan ring     = anchor joints",          C_HEEL_RING),
    ]
    for k, (s, c) in enumerate(legend):
        y = H_ - 14 - (len(legend)-1-k)*18
        txt(img, s, (W_-350, y), 0.32, c)


# ── Export JSON ───────────────────────────────────────────────────────────────

def export_json(data, derived, params, athlete):
    from export_lowerbody import (JOINT_NAMES, JOINT_DESCRIPTIONS, SKELETON_EDGES,
                                  MRIDULA_HIGHLIGHT)
    T   = data["T"]
    W, H = data["W"], data["H"]
    y_floor = data["y_floor"]

    def yup(p):
        q = p.copy(); q[..., 1] = y_floor - q[..., 1]; return q

    jidx = data["jidx"]
    J_all = data["J_all"]

    extra = derived
    joints_out = {}
    for jname in JOINT_NAMES:
        if jname in extra:
            pos = extra[jname]
        elif jname in jidx:
            pos = J_all[:, jidx[jname], :]
        else:
            continue
        pixels = []
        for t in range(T):
            px = project_pt(pos[t], data["cam_R"][t], data["cam_t"][t],
                            float(data["focal"][t]), W, H)
            pixels.append(list(px) if px else [None, None])
        joints_out[jname] = {"world_m": yup(pos).tolist(), "pixel_uv": pixels}

    out = {
        "athlete": athlete, "n_frames": T, "fps": 30.0,
        "tip_computation": data["mode"],
        "heel_tip_dist_m":      params["HEEL_TIP_DIST"],
        "heel_contact_post_m":  params["HEEL_CONTACT_POST"],
        "heel_contact_inf_m":   params["HEEL_CONTACT_INF"],
        "toe_dist_m":           params["TOE_DIST"],
        "coordinate_system": {
            "world_m": "Y-up, meters. Y=0 is floor.",
            "pixel_uv": "Origin=top-left. u=right, v=down.",
        },
        "joint_names": JOINT_NAMES,
        "joint_descriptions": JOINT_DESCRIPTIONS,
        "mridula_highlight": MRIDULA_HIGHLIGHT,
        "skeleton_edges": SKELETON_EDGES,
        "camera": {
            "image_width": W, "image_height": H,
            "focal_length_px": data["focal"].tolist(),
            "principal_point_px": [W/2, H/2],
            "cam_R": data["cam_R"].tolist(),
            "cam_t": data["cam_t"].tolist(),
            "y_floor_ydown": y_floor,
        },
        "joints": joints_out,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, f"{athlete}.json")
    with open(out_path, "w") as f: json.dump(out, f, indent=2)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Exported → {out_path}  ({size_mb:.1f} MB)")


# ── Save params back into export_lowerbody.py ─────────────────────────────────

def save_params(params):
    with open(EXPORT_PY) as f: src = f.read()
    replacements = {
        "HEEL_TIP_DIST":    f"{params['HEEL_TIP_DIST']:.4f}",
        "HEEL_CONTACT_POST": f"{params['HEEL_CONTACT_POST']:.4f}",
        "HEEL_CONTACT_INF":  f"{params['HEEL_CONTACT_INF']:.4f}",
        "TOE_DIST":          f"{params['TOE_DIST']:.4f}",
    }
    for key, val in replacements.items():
        src = re.sub(
            rf"^({key}\s*=\s*)[\d.]+",
            lambda m, v=val: m.group(1) + v,
            src, flags=re.MULTILINE
        )
    with open(EXPORT_PY, "w") as f: f.write(src)
    print(f"  Saved params to {EXPORT_PY}")
    for k, v in replacements.items():
        print(f"    {k} = {v}")


# ── Export overlay video ──────────────────────────────────────────────────────

def export_video(data, params, athlete, show_set, frames):
    OUT_DIR = os.path.join(HERE, "overlays")
    os.makedirs(OUT_DIR, exist_ok=True)
    tag    = "_".join(sorted(show_set)).replace("l_","").replace("r_","")[:40]
    out    = os.path.join(OUT_DIR, f"{athlete}_tuned_{tag}.mp4")
    W, H   = data["W"], data["H"]
    fps    = 30.0
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    T      = min(data["T"], len(frames))
    derived = recompute(data, params)
    print(f"  Rendering {T} frames → {out}")
    for fi in range(T):
        img        = frames[fi].copy()
        dpx        = get_pixels(derived, data, fi)
        apx        = get_anchor_pixels(data, fi)
        draw(img, fi, dpx, apx, params, True, show_set, W, H,
             athlete, T, data["mode"], None)
        writer.write(img)
    writer.release()
    print(f"  Done → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--athlete", default="Goree", choices=ATHLETES)
    ap.add_argument("--joints",  default="heel_contact,heel_tip,toe_tip,l_foot,r_foot,l_ball,r_ball",
                    help="Comma-separated joint name substrings to show.")
    args = ap.parse_args()

    # ── Load pkl ──────────────────────────────────────────────────────────────
    pkl_path = os.path.join(ROLLOUT_DIR, args.athlete, "mhr_kpts.pkl")
    rd_path  = os.path.join(ROLLOUT_DIR, args.athlete, "rollout.pkl")
    vid_path = os.path.join(VID_DIR, f"{args.athlete}.mp4")
    for p, lbl in [(pkl_path,"mhr_kpts.pkl"),(rd_path,"rollout.pkl"),(vid_path,"video")]:
        if not os.path.exists(p): sys.exit(f"ERROR: {lbl} not found: {p}")

    print(f"Loading {args.athlete} ...")
    with open(pkl_path,"rb") as f: mk = pickle.load(f)
    with open(rd_path, "rb") as f: rd = pickle.load(f)

    jnames = mk["joint_names"]
    J_all  = mk["all_joint_positions_world_m"]   # (T, J, 3) Y-DOWN, meters
    T      = J_all.shape[0]
    fi_map = {n: jnames.index(n) for n in jnames}

    cam_R  = rd["joint_cam_human_optim"]["cam_R"].astype(np.float64)
    cam_t  = rd["joint_cam_human_optim"]["cam_tvec"].astype(np.float64)
    focal  = rd["camera_intrinsics_used"]["focal_length_px"].astype(np.float64)
    W, H   = 1920, 1080

    has_bone = "l_foot_world_rot" in mk
    mode     = "bone-rotation" if has_bone else "geometric-fallback"
    print(f"  T={T}  mode={mode}")

    y_floor = float(np.max([J_all[:, fi_map[n], 1]
                             for n in ["l_foot","r_foot","l_ball","r_ball"]]))

    SKEL_JOINTS = ["root","l_upleg","l_lowleg","l_foot","l_talocrural","l_subtalar",
                   "l_transversetarsal","l_ball","r_upleg","r_lowleg","r_foot",
                   "r_talocrural","r_subtalar","r_transversetarsal","r_ball"]
    jidx = {n: fi_map[n] for n in SKEL_JOINTS if n in fi_map}

    data = {
        "lf":   J_all[:, jidx["l_foot"], :],
        "rf":   J_all[:, jidx["r_foot"], :],
        "lb":   J_all[:, jidx["l_ball"], :],
        "rb":   J_all[:, jidx["r_ball"], :],
        "ltmt": J_all[:, jidx["l_transversetarsal"], :],
        "rtmt": J_all[:, jidx["r_transversetarsal"], :],
        "has_bone": has_bone,
        "cam_R": cam_R, "cam_t": cam_t, "focal": focal,
        "W": W, "H": H, "T": T, "y_floor": y_floor,
        "J_all": J_all, "jidx": jidx, "mode": mode,
    }
    if has_bone:
        data["lf_rot"] = mk["l_foot_world_rot"]
        data["rf_rot"] = mk["r_foot_world_rot"]
        data["lb_rot"] = mk["l_ball_world_rot"]
        data["rb_rot"] = mk["r_ball_world_rot"]

    # ── Load video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(vid_path)
    frames = []
    for _ in range(T):
        ret, fr = cap.read()
        if not ret: break
        frames.append(fr)
    cap.release()
    T = len(frames)
    print(f"  Video: {T} frames ({W}×{H})")

    # ── Build joint filter set ────────────────────────────────────────────────
    ALL_DERIVED = ["l_heel_tip","r_heel_tip","l_heel_contact","r_heel_contact",
                   "l_toe_tip","r_toe_tip"]
    ANCHOR_NAMES = {"l_foot","r_foot","l_ball","r_ball"}
    ALL_NAMES    = ALL_DERIVED + list(ANCHOR_NAMES)

    tokens = [t.strip() for t in args.joints.split(",") if t.strip()]
    show_set = set()
    for tok in tokens:
        for n in ALL_NAMES:
            if tok in n: show_set.add(n)
    if not show_set: show_set = set(ALL_NAMES)
    print(f"  Showing: {sorted(show_set)}")

    # ── Initial params ────────────────────────────────────────────────────────
    params = {
        "HEEL_TIP_DIST":     0.055,
        "HEEL_CONTACT_POST": 0.050,
        "HEEL_CONTACT_INF":  0.110,
        "TOE_DIST":          0.035,
    }

    # ── Window ────────────────────────────────────────────────────────────────
    if not os.environ.get("DISPLAY"):
        sys.exit("No DISPLAY set.")

    fi           = T // 2
    show_labels  = True
    last_changed = None
    dirty        = True   # recompute needed

    derived     = recompute(data, params)
    derived_px  = get_pixels(derived, data, fi)
    anchor_px   = get_anchor_pixels(data, fi)

    win = f"tune_heel — {args.athlete}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(W, 1440), min(H, 900))

    def render():
        img = frames[fi].copy()
        draw(img, fi, derived_px, anchor_px, params, show_labels,
             show_set, W, H, args.athlete, T, mode, last_changed)
        cv2.imshow(win, img)

    render()
    print("\nControls: z/x c/v b/n m/, = tune (-/+) | ←/→ = frame | r=export | w=save-py | q=quit\n")

    while True:
        key = cv2.waitKey(0) & 0xFF
        prev_params = dict(params)
        last_changed = None

        # ── Frame navigation ──────────────────────────────────────────────────
        if   key in (83, ord('d')):        fi = min(fi+1, T-1)
        elif key in (81, ord('a')):        fi = max(fi-1, 0)
        elif key in (ord('+'),ord('=')):   fi = min(fi+10, T-1)
        elif key == ord('-'):              fi = max(fi-10, 0)
        elif ord('0') <= key <= ord('9'): fi = int((key-ord('0'))/10*(T-1))

        # ── Param tuning (bottom-row key pairs: left=minus, right=plus) ─────────
        elif key == ord('z'): params["HEEL_TIP_DIST"]     = max(0.005, params["HEEL_TIP_DIST"]     - STEP); last_changed="HEEL_TIP_DIST"
        elif key == ord('x'): params["HEEL_TIP_DIST"]     = params["HEEL_TIP_DIST"]     + STEP;             last_changed="HEEL_TIP_DIST"
        elif key == ord('c'): params["HEEL_CONTACT_POST"] = max(0.005, params["HEEL_CONTACT_POST"] - STEP); last_changed="HEEL_CONTACT_POST"
        elif key == ord('v'): params["HEEL_CONTACT_POST"] = params["HEEL_CONTACT_POST"] + STEP;             last_changed="HEEL_CONTACT_POST"
        elif key == ord('b'): params["HEEL_CONTACT_INF"]  = max(0.0,   params["HEEL_CONTACT_INF"]  - STEP); last_changed="HEEL_CONTACT_INF"
        elif key == ord('n'): params["HEEL_CONTACT_INF"]  = params["HEEL_CONTACT_INF"]  + STEP;             last_changed="HEEL_CONTACT_INF"
        elif key == ord('m'): params["TOE_DIST"]          = max(0.005, params["TOE_DIST"]          - STEP); last_changed="TOE_DIST"
        elif key == 44:       params["TOE_DIST"]          = params["TOE_DIST"]          + STEP;             last_changed="TOE_DIST"  # ','

        # ── Actions ───────────────────────────────────────────────────────────
        elif key == ord('l'): show_labels = not show_labels
        elif key == ord('e'):
            print("Exporting overlay video ...")
            export_video(data, params, args.athlete, show_set, frames)
        elif key == ord('r'):
            print("Exporting JSON ...")
            export_json(data, derived, params, args.athlete)
        elif key == ord('w'):
            save_params(params)
        elif key in (ord('q'), 27): break

        # recompute if params changed or frame changed
        if params != prev_params or last_changed:
            derived    = recompute(data, params)
            if last_changed:
                print(f"  {last_changed} = {params[last_changed]*100:.1f} cm")

        derived_px = get_pixels(derived, data, fi)
        anchor_px  = get_anchor_pixels(data, fi)
        render()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
