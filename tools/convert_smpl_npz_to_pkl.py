#!/usr/bin/env python3
"""
convert_smpl_npz_to_pkl.py — Convert an SMPL .npz model file to .pkl format.

The rollout optimizer loads the SMPL body model via smplx.create(), which accepts
either .npz or .pkl files. This script converts the official SMPL .npz download
to the .pkl format used in this pipeline.

Where to get the SMPL model
────────────────────────────
1. Register (free) at https://smpl.is.tue.mpg.de/
2. Download "SMPL for Python Users" → extract the zip
3. You will find files like:
     basicModel_neutral_lbs_10_207_0_v1.0.0.npz   (neutral)
     basicModel_m_lbs_10_207_0_v1.0.0.npz          (male)
     basicModel_f_lbs_10_207_0_v1.0.0.npz          (female)
   OR (newer release):
     SMPL_NEUTRAL.npz / SMPL_MALE.npz / SMPL_FEMALE.npz

Usage
─────
  python tools/convert_smpl_npz_to_pkl.py \
      --input  /path/to/SMPL_NEUTRAL.npz \
      --output /path/to/smplhub/smpl/SMPL_N_model_generate_from_npz.pkl

  # Convert all three genders at once:
  python tools/convert_smpl_npz_to_pkl.py \
      --input  /path/to/SMPL_NEUTRAL.npz \
      --output /path/to/smplhub/smpl/SMPL_N_model_generate_from_npz.pkl

  python tools/convert_smpl_npz_to_pkl.py \
      --input  /path/to/SMPL_MALE.npz \
      --output /path/to/smplhub/smpl/SMPL_M_model.pkl

  python tools/convert_smpl_npz_to_pkl.py \
      --input  /path/to/SMPL_FEMALE.npz \
      --output /path/to/smplhub/smpl/SMPL_F_model.pkl

Note: the pipeline only requires the NEUTRAL model (SMPL_N_model_generate_from_npz.pkl).
"""
import argparse
import os
import pickle
import numpy as np


def convert(input_path: str, output_path: str) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input not found: {input_path}")

    print(f"Loading  {input_path}")
    data = np.load(input_path, allow_pickle=True)
    d = {k: data[k] for k in data.files}

    expected_keys = {"J_regressor", "f", "kintree_table", "posedirs",
                     "shapedirs", "v_template", "weights"}
    missing = expected_keys - set(d.keys())
    if missing:
        raise ValueError(
            f"Input .npz is missing expected SMPL keys: {missing}\n"
            f"Found keys: {sorted(d.keys())}\n"
            f"Make sure you downloaded the SMPL model from https://smpl.is.tue.mpg.de/"
        )

    print(f"  Keys: {sorted(d.keys())}")
    print(f"  v_template shape: {d['v_template'].shape}  "
          f"(expected: (6890, 3) for SMPL)")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(d, f)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Saved    {output_path}  ({size_kb:.0f} KB)")
    print("Done.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input",  required=True, help="Path to SMPL .npz file")
    ap.add_argument("--output", required=True, help="Path for output .pkl file")
    args = ap.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
