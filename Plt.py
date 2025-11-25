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

SYMMETRIC_COLOR = True
# ============================================


# Custom scientific infrared-style colormap: purple → blue → cyan → yellow → white
SCI_IR_COLORMAP = LinearSegmentedColormap.from_list(
    "sci_ir",
    [
        (0.00, "#2d004b"),
        (0.25, "#1d3f72"),
        (0.50, "#38a8a6"),
        (0.75, "#eed54f"),
        (1.00, "#ffffff"),
    ],
)


def load_all_matrices(path):
    shape_pattern = re.compile(r"\((\d+),\s*(\d+)\)")
    matrices = []
    current_shape = None
    current_data = []

    with open(path, "r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if set(line) <= {"-", " "}:
                if current_shape is not None and current_data:
                    rows, cols = current_shape
                    required = rows * cols
                    arr = np.array(current_data[:required], dtype=float).reshape(rows, cols)
                    matrices.append(arr)
                current_shape = None
                current_data = []
                continue

            if line.startswith("Shape:"):
                m = shape_pattern.search(line)
                if m:
                    current_shape = (int(m.group(1)), int(m.group(2)))
                    current_data = []
                continue

            if line.startswith("Data:"):
                data_part = line.split("Data:", 1)[1].strip()
                for tok in data_part.split():
                    try:
                        current_data.append(float(tok))
                    except ValueError:
                        pass
                continue

            for tok in line.split():
                try:
                    current_data.append(float(tok))
                except ValueError:
                    pass

    if current_shape is not None and current_data:
        rows, cols = current_shape
        required = rows * cols
        arr = np.array(current_data[:required], dtype=float).reshape(rows, cols)
        matrices.append(arr)

    return matrices


def plot_matrix(mat, output_file, symmetric=True, dpi=300, title=None):
    plt.figure(figsize=(6, 5))

    if symmetric:
        vmax = float(np.max(np.abs(mat)))
        vmin = -vmax
    else:
        vmin, vmax = None, None

    im = plt.imshow(mat, vmin=vmin, vmax=vmax, aspect="equal", cmap=SCI_IR_COLORMAP)
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
