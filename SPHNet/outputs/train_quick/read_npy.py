# --*-- conding:utf-8 --*--
# @time:8/21/25 20:42
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:read_npy.py

import numpy as np
import glob
import os


if __name__ == '__main__':

    # Path to predictions folder
    pred_dir = "./predictions"
    output_file = os.path.join(pred_dir, "predictions.txt")

    # Get all .npy files
    files = sorted(glob.glob(os.path.join(pred_dir, "*.npy")))

    with open(output_file, "w") as f_out:
        for file in files:
            arr = np.load(file)
            f_out.write(f"File: {os.path.basename(file)}\n")
            f_out.write(f"Shape: {arr.shape}\n")
            f_out.write("Data:\n")
            np.savetxt(f_out, arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr, fmt="%.6f")
            f_out.write("\n" + "-"*60 + "\n\n")

    print(f"All predictions have been saved to: {output_file}")
