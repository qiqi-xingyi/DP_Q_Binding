# --*-- coding:utf-8 --*--
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

# All heatmaps will be saved here (directory will be auto-created)
OUTPUT_DIR = r"/Users/yuqizhang/Desktop/Code/DP_Quantum_binding/heatmaps"

SYMMETRIC_COLOR = True
# ============================================


def load_all_matrices(path):
    """
    Parse all matrices from the predictions file.
    Each matrix block is assumed to follow this pattern:

        Shape: (R, C)
        Data: x x x ...
        x x x ...
        ----  (separator line)

    A separator line is any line composed only of '-' and spaces.
    """

    shape_pattern = re.compile(r"\((\d+),\s*(\d+)\)")

    matrices = []
    current_shape = None
    current_data = []

    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # Separator: finalize current matrix
            if set(line) <= {"-", " "}:
                if current_shape is not None and current_data:
                    rows, cols = current_shape
                    required = rows * cols
                    if len(current_data) < required:
                        raise ValueError(
                            f"Insufficient data for matrix {current_shape}: "
                            f"got {len(current_data)} values."
                        )
                    arr = np.array(current_data[:required], dtype=float).reshape(rows, cols)
                    matrices.append(arr)

                current_shape = None
                current_data = []
                continue

            # Parse shape line
            if line.startswith("Shape:"):
                m = shape_pattern.search(line)
                if m:
                    rows = int(m.group(1))
                    cols = int(m.group(2))
                    current_shape = (rows, cols)
                    current_data = []
                continue

            # Parse data line beginning
            if line.startswith("Data:"):
                data_part = line.split("Data:", 1)[1].strip()
                if data_part:
                    for tok in data_part.split():
                        try:
                            current_data.append(float(tok))
                        except ValueError:
                            pass
                continue

            # Additional numeric lines
            for tok in line.split():
                try:
                    current_data.append(float(tok))
                except ValueError:
                    pass

    # Final matrix at EOF
    if current_shape is not None and current_data:
        rows, cols = current_shape
        required = rows * cols
        if len(current_data) < required:
            raise ValueError(
                f"Insufficient data for final matrix {current_shape}: "
                f"got {len(current_data)} values."
            )
        arr = np.array(current_data[:required], dtype=float).reshape(rows, cols)
        matrices.append(arr)

    if not matrices:
        raise ValueError("No matrices were parsed from the file.")

    return matrices


def plot_matrix(mat, output_file, symmetric=True, dpi=300, title=None):
    plt.figure(figsize=(6, 5))

    if symmetric:
        vmax = float(np.max(np.abs(mat)))
        vmin = -vmax
        im = plt.imshow(mat, vmin=vmin, vmax=vmax, aspect="equal")
    else:
        im = plt.imshow(mat, aspect="equal")

    plt.colorbar(im, fraction=0.046, pad=0.04)
    if title is None:
        title = "Molecular Matrix After Quantum-Chemical Modeling"
    plt.title(title)
    plt.xlabel("Column index")
    plt.ylabel("Row index")
    plt.tight_layout()

    out_path = Path(output_file)
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"Saved heatmap to: {out_path}")


def run():
    pred_path = Path(PREDICTIONS_FILE)
    if not pred_path.is_file():
        raise FileNotFoundError(f"File not found: {pred_path}")

    matrices = load_all_matrices(pred_path)
    print(f"Parsed {len(matrices)} matrices.")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx, mat in enumerate(matrices):
        out_file = out_dir / f"matrix_{idx:04d}.png"
        title = f"Molecular Matrix After Quantum-Chemical Modeling #{idx}"
        plot_matrix(mat, out_file, symmetric=SYMMETRIC_COLOR, title=title)


if __name__ == "__main__":
    run()
