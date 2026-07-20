#!/usr/bin/env python3
"""
Inspect the structure and contents of a .pkl file.

Recursively walks dicts, lists, tuples, and numpy arrays —
printing type, shape, dtype, and sample values at each level.

Usage:
    python inspect_pkl.py /home/nan/Desktop/NRMFOptim/sprintMesh2/out_sequence_smpl_rots_with_2d_kpts.pkl
    python inspect_pkl.py /path/to/file.pkl --verbose
    python inspect_pkl.py /path/to/file.pkl --depth 4
    python inspect_pkl.py /path/to/file.pkl --key smpl_parameters
"""

import argparse
import pickle
import numpy as np


# ─────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────

def _indent(depth):
    return "  " * depth


def _array_summary(arr, max_values=6):
    parts = [f"ndarray  shape={arr.shape}  dtype={arr.dtype}"]
    if arr.size > 0:
        parts.append(f"min/max={arr.min():.4g}/{arr.max():.4g}")
        flat = arr.ravel()
        sample = "  ".join(f"{v:.4g}" for v in flat[:max_values])
        tail = " ..." if arr.size > max_values else ""
        parts.append(f"sample=[{sample}{tail}]")
    return "  ".join(parts)


def _scalar_summary(val):
    t = type(val).__name__
    if isinstance(val, (int, float, bool)):
        return f"{t}  value={val}"
    if isinstance(val, str):
        preview = val[:80] + ("..." if len(val) > 80 else "")
        return f"str  len={len(val)}  value={repr(preview)}"
    return f"{t}  repr={repr(val)[:80]}"


# ─────────────────────────────────────────────
# Recursive walker
# ─────────────────────────────────────────────

def _walk(obj, depth, max_depth, verbose):
    pad = _indent(depth)

    if depth > max_depth:
        print(f"{pad}... (max depth reached)")
        return

    if isinstance(obj, dict):
        print(f"{pad}dict  ({len(obj)} keys)")
        for k, v in obj.items():
            print(f"{pad}  [{repr(k)}]")
            _walk(v, depth + 2, max_depth, verbose)

    elif isinstance(obj, (list, tuple)):
        kind = type(obj).__name__
        print(f"{pad}{kind}  len={len(obj)}")
        limit = len(obj) if verbose else min(len(obj), 5)
        for i in range(limit):
            print(f"{pad}  [{i}]")
            _walk(obj[i], depth + 2, max_depth, verbose)
        if not verbose and len(obj) > 5:
            print(f"{pad}  ... ({len(obj) - 5} more items, use --verbose to show all)")

    elif isinstance(obj, np.ndarray):
        print(f"{pad}{_array_summary(obj)}")
        if verbose and obj.dtype == object and obj.size > 0:
            print(f"{pad}  object entries (first 3):")
            for i, item in enumerate(obj.flat[:3]):
                print(f"{pad}    [{i}] type={type(item).__name__}  {repr(item)[:60]}")

    else:
        print(f"{pad}{_scalar_summary(obj)}")


# ─────────────────────────────────────────────
# Main inspect function
# ─────────────────────────────────────────────

def inspect_pkl(path, max_depth=6, verbose=False, key=None):
    print(f"\nInspecting: {path}\n")

    with open(path, "rb") as f:
        data = pickle.load(f)
    pdb.set_trace()

    # Optionally drill into a specific top-level key
    if key is not None:
        if not isinstance(data, dict) or key not in data:
            available = list(data.keys()) if isinstance(data, dict) else "(not a dict)"
            print(f"Key '{key}' not found. Available: {available}")
            return
        print(f"  Showing key: '{key}'\n")
        data = data[key]

    _walk(data, depth=0, max_depth=max_depth, verbose=verbose)
    print("\nDone.")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inspect .pkl file structure")
    parser.add_argument("pkl_path", type=str, help="Path to the .pkl file")
    parser.add_argument("--depth",   type=int, default=6,
                        help="Maximum recursion depth (default: 6)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show all list items and object-array entries")
    parser.add_argument("--key",     type=str, default=None,
                        help="Drill into a specific top-level dict key")
    args = parser.parse_args()

    inspect_pkl(args.pkl_path, max_depth=args.depth, verbose=args.verbose, key=args.key)


if __name__ == "__main__":
    main()
