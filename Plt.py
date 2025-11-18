# --*-- conding:utf-8 --*--
# @time:11/18/25 11:09
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:Plt.py
import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# =======================================================
# CONFIG — YOU ONLY NEED TO CHANGE THIS PATH
# =======================================================
PREDICTIONS_FILE = r"SPHNet/outputs/train_quick/predictions/predictions.txt"

# Optional: choose output directory (default: same folder + /plots/)
OUTPUT_DIR = None

# Whether to use symmetric color scale around zero
SYMMETRIC_COLOR = True


def parse_predictions_file(path):
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
                if current_name is not None and current_shape is not None and current_data:
                    arr = np.array(current_data, dtype=float).reshape(current_shape)
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

            elif line.startswith("Data:"):
                data_part = line.split("Data:", 1)[1].strip()
                if data_part:
                    current_data.extend(data_part.split())

            else:
                current_data.extend(line.split())

        if current_name is not None and current_shape is not None and current_data:
            arr = np.array(current_data, dtype=float).reshape(current_shape)
            matrices.append((current_name, arr))

    return matrices


def plot_matrix(mat, name, out_dir, symmetric=True, dpi=300):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(name).name
    if stem.endswith(".npy"):
        stem = stem[:-4]

    plt.figure(figsize=(6, 5))

    if symmetric:
        vmax = np.max(np.abs(mat))
        vmin = -vmax
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


def run():
    pred_path = Path(PREDICTIONS_FILE)
    if not pred_path.is_file():
        raise FileNotFoundError(f"Could not find predictions file: {pred_path}")

    if OUTPUT_DIR is None:
        out_dir = pred_path.parent / "plots"
    else:
        out_dir = Path(OUTPUT_DIR)

    matrices = parse_predictions_file(pred_path)

    print(f"Parsed {len(matrices)} matrices from {pred_path}")

    for name, mat in matrices:
        plot_matrix(
            mat,
            name=name,
            out_dir=out_dir,
            symmetric=SYMMETRIC_COLOR,
        )


if __name__ == "__main__":
    run()
