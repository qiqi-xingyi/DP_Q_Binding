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

# optional import: torch_geometric Data support
try:
    from torch_geometric.data import Data as TGData
except Exception:
    TGData = None  # handle gracefully if torch_geometric is not present


def parse_args():
    p = argparse.ArgumentParser(
        description="SPHNet inference: load ckpt, run predictions on MDB/LMDB dataset, save Hamiltonians"
    )
    p.add_argument("--job_id", type=str, default="train_quick",
                   help="Training job_id to locate outputs/<job_id>/ for ckpt discovery")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Optional explicit ckpt path; if omitted, latest ckpt in outputs/<job_id>/ is used")
    p.add_argument("--outputs_dir", type=str, default="outputs",
                   help="Root outputs directory (default: outputs)")
    p.add_argument("--save_dirname", type=str, default="predictions",
                   help="Subdir under outputs/<job_id>/ to save predictions (default: predictions)")

    p.add_argument("--dataset_path", type=str, default="./example_data/data.mdb",
                   help="Path to MDB/LMDB dataset file")
    p.add_argument("--data_name", type=str, default="custom",
                   help="Dataset name (must match training)")
    p.add_argument("--basis", type=str, default="def2-svp",
                   help="Basis (must match training)")
    p.add_argument("--index_path", type=str, default=".",
                   help="Path to index .pt if your split requires it")

    p.add_argument("--batch_size", type=int, default=1, help="Inference batch size")
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers; set 0 if you faced multiprocessing issues")
    p.add_argument("--split", type=str, default="test", choices=["test", "val", "train"],
                   help="Which split to run predictions on")
    p.add_argument("--precision", type=str, default="32", choices=["16", "32", "64"],
                   help="Precision for Lightning Trainer")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                   help="Device selection")
    p.add_argument("--ham_key", type=str, default=None,
                   help="Optional: explicit key/attr name to read Hamiltonian from model output "
                        "(e.g. 'hami_pred', 'hami', 'fock'). Overrides auto-detection.")
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
    """Return (accelerator, devices) for Lightning Trainer."""
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


def extract_hamiltonian(out, ham_key=None):
    """
    Extract Hamiltonian tensor from model output.
    Accepts:
      - dict: looks up ham_key (if given) then common keys
      - torch_geometric.data.Data: looks up ham_key, then common attrs, then auto-detects square matrix
      - Tensor: returned as-is
      - 1-item list/tuple: unwrapped and processed
    Returns a torch.Tensor with shape [n, n] or [B, n, n].
    """
    # unwrap singletons
    if isinstance(out, (list, tuple)) and len(out) == 1:
        return extract_hamiltonian(out[0], ham_key=ham_key)

    # dict case
    if isinstance(out, dict):
        if ham_key and ham_key in out:
            v = out[ham_key]
            if torch.is_tensor(v):
                return v
            raise RuntimeError(f"Value at ham_key='{ham_key}' is not a tensor: {type(v)}")

        for k in ("hami_pred", "hami", "pred", "H", "hamiltonian", "hamiltonian_pred", "fock", "fock_pred"):
            if k in out and torch.is_tensor(out[k]):
                return out[k]
        raise RuntimeError(f"Hamiltonian not found in dict; keys={list(out.keys())}")

    # torch_geometric Data case
    if TGData is not None and isinstance(out, TGData):
        if ham_key and hasattr(out, ham_key):
            v = getattr(out, ham_key)
            if torch.is_tensor(v):
                return v
            raise RuntimeError(f"Attr at ham_key='{ham_key}' is not a tensor: {type(v)}")

        # try common attribute names
        for k in ("hami_pred", "hami", "pred", "H", "hamiltonian", "hamiltonian_pred", "fock", "fock_pred"):
            if hasattr(out, k):
                v = getattr(out, k)
                if torch.is_tensor(v):
                    return v

        # auto-detect a square (or batched square) tensor among attributes
        candidates = []
        # iterate over tensor attrs of TGData (avoid private attrs)
        for k, v in out.__dict__.items():
            if k.startswith("_"):
                continue
            if torch.is_tensor(v):
                if v.ndim == 2 and v.shape[0] == v.shape[1]:
                    candidates.append((k, v))
                elif v.ndim == 3 and v.shape[-1] == v.shape[-2]:
                    candidates.append((k, v))

        if len(candidates) == 1:
            return candidates[0][1]
        elif len(candidates) > 1:
            names = [k for k, _ in candidates]
            raise RuntimeError(
                f"Multiple square-matrix candidates found in torch_geometric.Data: {names}. "
                f"Please specify --ham_key to disambiguate."
            )
        else:
            tensor_names = [k for k, v in out.__dict__.items() if torch.is_tensor(v)]
            raise RuntimeError(
                "No square-matrix-like tensor found in torch_geometric.Data. "
                f"Tensor attrs: {tensor_names}"
            )

    # plain tensor
    if torch.is_tensor(out):
        return out

    raise RuntimeError(f"Unsupported output type: {type(out)}")


def save_predictions(preds, save_root, ham_key=None):
    """
    Save Hamiltonians from a list of prediction outputs.
    Each matrix is saved as .npy and indexed in manifest.json.
    """
    os.makedirs(save_root, exist_ok=True)

    manifest = []
    idx = 0
    for out in preds:
        outs = out if isinstance(out, (list, tuple)) else [out]
        for o in outs:
            H = extract_hamiltonian(o, ham_key=ham_key)
            if torch.is_tensor(H):
                H = H.detach().cpu().numpy()

            if H.ndim == 3:
                for i in range(H.shape[0]):
                    fn = f"sample_{idx:06d}_H.npy"
                    np.save(os.path.join(save_root, fn), H[i])
                    manifest.append({"index": idx, "file": fn, "shape": list(H[i].shape)})
                    idx += 1
            elif H.ndim == 2:
                fn = f"sample_{idx:06d}_H.npy"
                np.save(os.path.join(save_root, fn), H)
                manifest.append({"index": idx, "file": fn, "shape": list(H.shape)})
                idx += 1
            else:
                raise RuntimeError(f"Unsupported Hamiltonian shape: {H.shape}")

    with open(os.path.join(save_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] Saved {len(manifest)} Hamiltonian matrices to {save_root}")
    print(f"[OK] Manifest: {os.path.join(save_root, 'manifest.json')}")


def main():
    args = parse_args()
    cfg = build_cfg(args)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # device
    accelerator, devices = pick_device(args)

    # locate ckpt
    if args.ckpt:
        ckpt_path = args.ckpt
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    else:
        ckpt_path = get_latest_ckpt(cfg.log_dir)
        if ckpt_path is None:
            cands = sorted(glob.glob(os.path.join(cfg.log_dir, "*.ckpt")))
            ckpt_path = cands[-1] if cands else None
        if ckpt_path is None:
            raise FileNotFoundError(f"No .ckpt found in {cfg.log_dir}")

    print(f"[info] Using ckpt: {ckpt_path}")
    print(f"[info] Using dataset: {cfg.dataset_path} ({cfg.data_name}, basis={cfg.basis})")
    print(f"[info] Predictions will be saved to: {cfg.log_dir}/{args.save_dirname}")

    # data
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

    # model
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

    save_root = os.path.join(cfg.log_dir, args.save_dirname)
    save_predictions(preds, save_root, ham_key=args.ham_key)


if __name__ == "__main__":
    main()

