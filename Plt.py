# --*-- coding:utf-8 --*--
# @time:11/18/25 11:09
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:Plt.py

import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ================== CONFIG ==================
PREDICTIONS_FILE = r"/Users/yuqizhang/Desktop/Code/DP_Quantum_binding/SPHNet/outputs/train_quick/predictions/predictions.txt"

OUTPUT_DIR = r"/Users/yuqizhang/Desktop/Code/DP_Quantum_binding/heatmaps"
# ============================================

# Custom infrared-style colormap: purple -> yellow -> white
IR_COLORMAP = LinearSegmentedColormap.from_list(
    "infrared_pyw",
    [
        (0.0, "purple"),
        (0.5, "yellow"),
        (1.0, "white"),
    ],
)


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


def normalize_matrix(mat):
    """
    Normalize a matrix to [0, 1].
    If the matrix is constant, return all zeros.
    """
    mat = np.asarray(mat, dtype=float)
    vmin = np.min(mat)
    vmax = np.max(mat)
    if vmax == vmin:
        norm = np.zeros_like(mat)
    else:
        norm = (mat - vmin) / (vmax - vmin)
    return norm


def plot_matrix(mat, output_file, dpi=300, title=None):
    """
    Plot a single normalized matrix with infrared-style colormap.
    """
    norm_mat = normalize_matrix(mat)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(norm_mat, vmin=0.0, vmax=1.0, aspect="equal", cmap=IR_COLORMAP)

    plt.colorbar(im, fraction=0.046, pad=0.04)
    if title is None:
        title = "Molecular Matrix After Quantum-Chemical Modeling (Normalized)"
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
        title = f"Molecular Matrix After Quantum-Chemical Modeling (Normalized) #{idx}"
        plot_matrix(mat, out_file, title=title)


if __name__ == "__main__":
    run()
