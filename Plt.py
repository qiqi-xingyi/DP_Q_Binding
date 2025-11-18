# --*-- conding:utf-8 --*--
# @time:11/18/25 11:09
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:Plt.py

import argparse
import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_predictions_file(path):
    """
    Parse predictions.txt-like file into a list of (name, array) pairs.
    """
    matrices = []
    current_name = None
    current_shape = None
    current_data = []

    shape_pattern = re.compile(r"\((\d+),\s*(\d+)\)")

    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("File:"):
                # flush previous matrix
                if current_name is not None and current_shape is not None and current_data:
                    arr = np.array(current_data, dtype=float)
                    arr = arr.reshape(current_shape)
                    matrices.append((current_name, arr))
                    current_data = []

                current_name = line.split("File:", 1)[1].strip()
                current_shape = None

            elif line.startswith("Shape:"):
                m = shape_pattern.search(line)
                if m:
                    rows = int(m.group(1))
                    cols = int(m.group(2))
                    current_shape = (rows, cols)
                else:
                    raise ValueError(f"Could not parse shape from line: {line}")

            elif line.startswith("Data:"):
                # data values may start from the same line or next lines
                data_part = line.split("Data:", 1)[1].strip()
                if data_part:
                    current_data.extend(data_part.split())

            else:
                # numeric data line
                current_data.extend(line.split())

        # flush last matrix
        if current_name is not None and current_shape is not None and current_data:
            arr = np.array(current_data, dtype=float)
            arr = arr.reshape(current_shape)
            matrices.append((current_name, arr))

    return matrices


def plot_matrix(mat, name, out_dir, symmetric=True, dpi=300):
    """
    Plot a single matrix as a heatmap and save to out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # derive a clean file stem from the original name
    stem = Path(name).name
    if stem.endswith(".npy"):
        stem = stem[:-4]

    plt.figure(figsize=(6, 5))

    if symmetric:
        vmax = np.max(np.abs(mat))
        if vmax == 0:
            vmin, vmax = -1.0, 1.0
        else:
            vmin, vmax = -vmax, vmax
        im = plt.imshow(mat, vmin=vmin, vmax=vmax, aspect="equal")
    else:
        im = plt.imshow(mat, aspect="equal")

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(stem)
    plt.xlabel("Column index")
    plt.ylabel("Row index")
    plt.tight_layout()

    out_path = out_dir / f"{stem}.png"
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot matrices from SPHNet predictions.txt as heatmaps."
    )
    parser.add_argument(
        "predictions",
        type=str,
        help="Path to predictions.txt (e.g., SPHNet/outputs/train_quick/predictions/predictions.txt)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save plots. Default: <predictions_dir>/plots",
    )
    parser.add_argument(
        "--no-symmetric",
        action="store_true",
        help="Disable symmetric color scale around zero.",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.is_file():
        raise FileNotFoundError(f"File not found: {pred_path}")

    if args.out_dir is None:
        out_dir = pred_path.parent / "plots"
    else:
        out_dir = Path(args.out_dir)

    matrices = parse_predictions_file(pred_path)
    if not matrices:
        print("No matrices parsed from file.")
        return

    print(f"Parsed {len(matrices)} matrices from {pred_path}")

    for name, mat in matrices:
        plot_matrix(
            mat,
            name=name,
            out_dir=out_dir,
            symmetric=not args.no_symmetric,
        )


if __name__ == "__main__":
    main()
