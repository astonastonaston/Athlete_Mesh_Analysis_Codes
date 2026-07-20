#!/usr/bin/env python3
"""
extract_frames_for_conversion.py

For one athlete, read per-frame pkl files from pkl_outputs/<athlete>/pkl_files/,
select the target detection (largest bbox by default), and write individual
per-frame pkl files that the MHR→SMPL conversion script can consume.

The conversion script expects each file to be a single-element list  [ {det_dict} ]
or a plain dict.  Our source pkls have N detections per frame, so this step
isolates the target person.

Usage
-----
python tools/extract_frames_for_conversion.py \
    --in_dir  outputs/pkl_outputs/Amari/pkl_files \
    --out_dir /tmp/extracted/Amari \
    --strategy largest_bbox

# Fixed index (e.g. always pick detection 2 for Bishop):
python tools/extract_frames_for_conversion.py \
    --in_dir  outputs/pkl_outputs/Bishop/pkl_files \
    --out_dir /tmp/extracted/Bishop \
    --strategy 2
"""

import argparse, os, pickle, sys
import numpy as np


def _bbox_areas(dets):
    """Return list of (area, original_index) sorted by area descending."""
    areas = []
    for i, det in enumerate(dets):
        b = np.asarray(det["bbox"]).flatten()
        a = float(b[2] - b[0]) * float(b[3] - b[1])
        areas.append((a, i))
    areas.sort(reverse=True)
    return areas  # [(area, det_idx), ...]


def select_detection(dets, strategy):
    """
    strategy options:
      'largest_bbox'      → detection with the largest bbox (rank 1)
      'rank:N'            → Nth largest bbox (N=1,2,3,…)
      '<integer>'         → fixed detection index (0-based, as output by MHR)
    """
    if len(dets) == 0:
        return None

    if strategy == "largest_bbox" or strategy == "rank:1":
        _, best_i = _bbox_areas(dets)[0]
        return dets[best_i]

    if strategy.startswith("rank:"):
        n = int(strategy.split(":")[1])
        ranked = _bbox_areas(dets)
        pick = ranked[min(n - 1, len(ranked) - 1)]
        return dets[pick[1]]

    # Fixed integer index
    idx = int(strategy)
    return dets[min(idx, len(dets) - 1)]


def main():
    p = argparse.ArgumentParser(
        description="Extract target-athlete detection from per-frame multi-person pkls.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--in_dir",   required=True,
                   help="Directory of per-frame pkl files (pkl_outputs/<Athlete>/pkl_files).")
    p.add_argument("--out_dir",  required=True,
                   help="Output directory for single-detection per-frame pkls.")
    p.add_argument("--strategy", default="largest_bbox",
                   help="'largest_bbox' or integer index.")
    p.add_argument("--skip_existing", type=int, default=1)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    strategy = (args.strategy if args.strategy == "largest_bbox"
                else int(args.strategy))

    fns = sorted(
        f for f in os.listdir(args.in_dir)
        if f.endswith(".pkl") and f.replace(".pkl", "").isdigit()
    )
    if not fns:
        print(f"[ERROR] No numeric pkl files in {args.in_dir}")
        sys.exit(1)

    skipped = extracted = 0
    for fn in fns:
        out_path = os.path.join(args.out_dir, fn)
        if args.skip_existing and os.path.isfile(out_path):
            skipped += 1
            continue

        with open(os.path.join(args.in_dir, fn), "rb") as f:
            dets = pickle.load(f)

        det = select_detection(dets, strategy)
        if det is None:
            print(f"  [warn] {fn}: no detections, skipping")
            continue

        # Save as single-element list — the format convert_mhr_to_smpl expects
        with open(out_path, "wb") as f:
            pickle.dump([det], f, protocol=4)
        extracted += 1

    print(f"[done] extracted={extracted}  skip_existing={skipped}  → {args.out_dir}")


if __name__ == "__main__":
    main()
