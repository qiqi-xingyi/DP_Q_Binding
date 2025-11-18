# --*-- conding:utf-8 --*--
# @time:11/18/25 11:09
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:Plt.py

import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ================== CONFIG ==================
PREDICTIONS_FILE = r"/Users/yuqizhang/Desktop/Code/DP_Quantum_binding/SPHNet/outputs/train_quick/predictions/predictions.txt"
OUTPUT_IMAGE = r"/Users/yuqizhang/Desktop/Code/DP_Quantum_binding/first_matrix.png"
SYMMETRIC_COLOR = True
# ============================================


def load_first_matrix(path):
    current_shape = None
    current_data = []

    shape_pattern = re.compile(r"\((\d+),\s*(\d+)\)")

    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # Stop at the first separator — only first matrix needed
            if set(line) <= {"-", " "}:
                break

            if line.startswith("Shape:"):
                m = shape_pattern.search(line)
                if m:
                    rows = int(m.group(1))
                    cols = int(m.group(2))
                    current_shape = (rows, cols)
                continue

            if line.startswith("Data:"):
                data_part = line.split("Data:", 1)[1].strip()
                if data_part:
                    for tok in data_part.split():
                        current_data.append(float(tok))
                continue

            # Additional numeric lines
            for tok in line.split():
                try:
                    current_data.append(float(tok))
                except ValueError:
                    pass

    if current_shape is None:
        raise ValueError("No matrix shape found in file.")

    rows, cols = current_shape
    arr = np.array(current_data, dtype=float).reshape((rows, cols))
    return arr


def plot_first_matrix(mat, output_file, symmetric=True, dpi=300):
    plt.figure(figsize=(6, 5))

    if symmetric:
        vmax = float(np.max(np.abs(mat)))
        vmin = -vmax
        im = plt.imshow(mat, vmin=vmin, vmax=vmax, aspect="equal")
    else:
        im = plt.imshow(mat, aspect="equal")

    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Molecular Matrix After Quantum-Chemical Modeling")
    plt.xlabel("Column index")
    plt.ylabel("Row index")
    plt.tight_layout()

    out_path = Path(output_file)
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"Saved first matrix heatmap to: {out_path}")


def run():
    pred_path = Path(PREDICTIONS_FILE)
    if not pred_path.is_file():
        raise FileNotFoundError(f"File not found: {pred_path}")

    mat = load_first_matrix(pred_path)
    plot_first_matrix(mat, OUTPUT_IMAGE, symmetric=SYMMETRIC_COLOR)


if __name__ == "__main__":
    run()
