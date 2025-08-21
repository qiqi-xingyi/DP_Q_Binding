# --*-- conding:utf-8 --*--
# @time:8/21/25 17:22
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:predict.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

# add src to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.utility.hydra_config import Config
from src.training.data import DataModule
from src.training.module import LNNP
from src.training.logger import get_latest_ckpt


def parse_args():
    p = argparse.ArgumentParser(description="SPHNet inference script: load ckpt, run predictions, and save Hamiltonians")
    p.add_argument("--job_id", type=str, default="train_quick",
                   help="Job ID used during training (to locate outputs/<job_id>/)")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Optional: explicit ckpt path; if not set, the latest ckpt from outputs/<job_id>/ will be used")
    p.add_argument("--outputs_dir", type=str, default="outputs",
                   help="Training outputs root directory")
    p.add_argument("--save_dirname", type=str, default="predictions",
                   help="Directory name for saving predictions under outputs/<job_id>/")
    p.add_argument("--dataset_path", type=str, default="./example_data/data.mdb",
                   help="LMDB/MDB dataset file")
    p.add_argument("--data_name", type=str, default="custom",
                   help="Dataset name (must match training config)")
    p.add_argument("--basis", type=str, default="def2-svp",
                   help="Basis set (must match training config)")
    p.add_argument("--index_path", type=str, default=".",
                   help="Path to index .pt file if needed by dataset")
    p.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers; set 0 if you face multiprocessing issues")
    p.add_argument("--split", type=str, default="test", choices=["test", "val", "train"],
                   help="Which split to run predictions on")
    p.add_argument("--precision", type=str, default="32", choices=["16", "32", "64"],
                   help="Precision mode for Lightning trainer")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                   help="Device selection")
    return p.parse_args()


def build_cfg(args):
    cfg = OmegaConf.structured(Config)
    cfg.data_name = args.data_name
    cfg.basis = args.basis
    cfg.dataset_path = args.dataset_path
    cfg.index_path = args.index_path
    cfg.batch_size = args.batch_size
    cfg.inference_batch_size = args.batch_size
    cfg.dataloader_num_workers = args.num_workers
    cfg.enable_hami = True
    cfg.ngpus = 1
    cfg.num_nodes = 1
    cfg.precision = args.precision
    cfg.job_id = args.job_id
    cfg.ckpt_path = args.outputs_dir
    cfg.log_dir = os.path.join(args.outputs_dir, args.job_id)
    return cfg


def pick_device(args):
    if args.device == "cpu":
        return "cpu", 0
    if args.device == "cuda":
        if torch.cuda.is_available():
            return "gpu", 1
        else:
            print("[warn] CUDA not available, falling back to CPU")
            return "cpu", 0
    # auto
    if torch.cuda.is_available():
        return "gpu", 1
    return "cpu", 0


def extract_H_from_output(out):
    """
    Extract Hamiltonian predictions from model outputs.
    - If dict: try common keys 'hami_pred', 'hami', 'pred'.
    - If tensor: return directly.
    """
    if isinstance(out, dict):
        for k in ("hami_pred", "hami", "pred"):
            if k in out:
                return out[k]
        raise RuntimeError(f"Hamiltonian not found in model output keys={list(out.keys())}")
    elif torch.is_tensor(out):
        return out
    else:
        raise RuntimeError(f"Unsupported output type: {type(out)}")


def main():
    args = parse_args()
    cfg = build_cfg(args)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # device
    accelerator, devices = pick_device(args)

    # locate ckpt
    if args.ckpt:
        ckpt_path = args.ckpt
        assert os.path.isfile(ckpt_path), f"ckpt not found: {ckpt_path}"
    else:
        ckpt_path = get_latest_ckpt(cfg.log_dir)
        if ckpt_path is None:
            cands = sorted(glob.glob(os.path.join(cfg.log_dir, "*.ckpt")))
            ckpt_path = cands[-1] if cands else None
        if ckpt_path is None:
            raise FileNotFoundError(f"No ckpt found in {cfg.log_dir}")

    print(f"[info] Using ckpt: {ckpt_path}")
    print(f"[info] Using dataset: {cfg.dataset_path} ({cfg.data_name}, basis={cfg.basis})")
    print(f"[info] Predictions will be saved to: {cfg.log_dir}/{args.save_dirname}")

    # Data
    dm = DataModule(cfg)
    if args.split == "test":
        dm.setup("test")
        loader = dm.test_dataloader()
    elif args.split == "val":
        dm.setup("val")
        loader = dm.val_dataloader()
    else:
        dm.setup("fit")
        loader = dm.train_dataloader()

    # Model
    if cfg.precision == "32":
        torch.set_float32_matmul_precision("high")
    model = LNNP.load_from_checkpoint(ckpt_path, config=cfg, strict=False)
    model.eval()

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        accelerator=accelerator,
        devices=devices,
        precision=cfg.precision,
        num_sanity_val_steps=0,
    )

    preds = trainer.predict(model, dataloaders=loader)

    # save predictions
    save_root = os.path.join(cfg.log_dir, args.save_dirname)
    os.makedirs(save_root, exist_ok=True)

    manifest = []
    idx = 0
    for batch_idx, out in enumerate(preds):
        outs = out if isinstance(out, (list, tuple)) else [out]
        for o in outs:
            H = extract_H_from_output(o)
            if torch.is_tensor(H):
                H = H.detach().cpu().numpy()

            if H.ndim == 3:  # batch of matrices
                for i in range(H.shape[0]):
                    fn = f"sample_{idx:06d}_H.npy"
                    np.save(os.path.join(save_root, fn), H[i])
                    manifest.append({"index": idx, "file": fn, "shape": list(H[i].shape)})
                    idx += 1
            elif H.ndim == 2:  # single matrix
                fn = f"sample_{idx:06d}_H.npy"
                np.save(os.path.join(save_root, fn), H)
                manifest.append({"index": idx, "file": fn, "shape": list(H.shape)})
                idx += 1
            else:
                raise RuntimeError(f"Unsupported shape: {H.shape}")

    with open(os.path.join(save_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Saved {len(manifest)} Hamiltonian matrices to {save_root}")
    print(f"[OK] Manifest file: {os.path.join(save_root, 'manifest.json')}")


if __name__ == "__main__":
    main()
