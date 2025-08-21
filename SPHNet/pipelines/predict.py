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
import argparse
from datetime import datetime

import torch
import pytorch_lightning as pl

# Add src in root folder
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(cur_dir, ".."))

from omegaconf import OmegaConf
from src.utility.hydra_config import Config
from src.training.data import DataModule
from src.training.module import LNNP
from src.training.logger import get_latest_ckpt

try:
    from torch_geometric.data import Data as TGData
except Exception:
    TGData = None


def parse_args():
    p = argparse.ArgumentParser(description="Inspect-only predictor: print keys and shapes of model outputs.")
    p.add_argument("--ckpt", type=str, default=None,
                   help="Path to a specific checkpoint. If not set, use the latest .ckpt under outputs/<job_id>/")
    p.add_argument("--job_id", type=str, default="train_quick",
                   help="Job ID (used to locate outputs/<job_id>/ if --ckpt is not given).")
    p.add_argument("--dataset_path", type=str, required=True,
                   help="Path to dataset file (e.g., ./example_data/data.mdb or .lmdb).")
    p.add_argument("--data_name", type=str, default="custom",
                   help="Dataset split name used by the repo (e.g., custom, qh9_stable_iid, ...).")
    p.add_argument("--basis", type=str, default="def2-svp",
                   help="Basis string used by the repo (e.g., def2-svp).")

    p.add_argument("--num_workers", type=int, default=0,
                   help="num_workers for predict dataloader to avoid segfaults on some systems.")
    p.add_argument("--device", type=str, default="cuda",
                   choices=["cuda", "cpu"], help="Accelerator hint. Trainer may still auto-select GPUs.")

    p.add_argument("--inspect_limit", type=int, default=3,
                   help="How many prediction items to print (default: 3).")
    p.add_argument("--print_values", action="store_true",
                   help="If set, also prints small tensors' values (<= 10x10) to stdout.")
    p.add_argument("--seed", type=int, default=123, help="Random seed.")

    return p.parse_args()


def load_config_and_patch(args):
    """Create a config object compatible with the repo's training code."""
    schema = OmegaConf.structured(Config)
    cfg = OmegaConf.merge(schema, {})  # start from defaults

    # Minimal fields needed for DataModule and model init
    cfg.job_id = args.job_id
    cfg.log_dir = os.path.join("outputs", cfg.job_id)
    cfg.ckpt_path = "outputs"
    cfg.data_name = args.data_name
    cfg.basis = args.basis
    cfg.dataset_path = args.dataset_path

    # Inference-related knobs
    cfg.inference_batch_size = 1 if cfg.get("inference_batch_size", None) is None else cfg.inference_batch_size
    cfg.dataloader_num_workers = args.num_workers
    cfg.ngpus = 1
    cfg.num_nodes = 1
    cfg.precision = cfg.get("precision", "32")
    cfg.num_sanity_val_steps = 0
    cfg.check_val_every_n_epoch = 1

    # Disable W&B by default to avoid login prompts during inspection
    if "wandb" not in cfg:
        cfg.wandb = {}
    cfg.wandb["open"] = False

    return cfg


def pretty_print_tensor(name, t, print_values=False, max_print=10):
    """Print tensor name, shape, dtype, min/max; optionally the values if small."""
    t_cpu = t.detach().cpu()
    desc = f"- {name}: Tensor shape={tuple(t_cpu.shape)}, dtype={t_cpu.dtype}"
    try:
        tmin = float(t_cpu.min())
        tmax = float(t_cpu.max())
        desc += f", min={tmin:.6g}, max={tmax:.6g}"
    except Exception:
        pass
    print(desc)
    if print_values:
        # Only print if it's reasonably small
        numel = t_cpu.numel()
        if t_cpu.ndim <= 2 and t_cpu.shape[0] <= max_print and (t_cpu.ndim == 1 or t_cpu.shape[-1] <= max_print) and numel <= max_print * max_print:
            print(t_cpu)
        else:
            print("  (values suppressed; set --print_values for small tensors only)")


def debug_print_output(idx, out, print_values=False):
    """
    Print all tensor fields from a prediction output (dict/TGData/tensor),
    including shapes and basic stats.
    """
    print(f"\n[inspect] >>> Item #{idx}")
    # Unwrap a 1-item list/tuple for readability
    if isinstance(out, (list, tuple)) and len(out) == 1:
        out = out[0]

    if torch.is_tensor(out):
        pretty_print_tensor("tensor", out, print_values=print_values)
        return

    if isinstance(out, dict):
        print(f"[inspect] type=dict, keys={list(out.keys())}")
        for k, v in out.items():
            if torch.is_tensor(v):
                pretty_print_tensor(k, v, print_values=print_values)
            else:
                print(f"- {k}: {type(v)}")
        return

    if TGData is not None and isinstance(out, TGData):
        ks = list(out.keys())
        print(f"[inspect] type=torch_geometric.data.Data, keys={ks}")
        for k in ks:
            v = out[k]
            if torch.is_tensor(v):
                pretty_print_tensor(k, v, print_values=print_values)
            else:
                print(f"- {k}: {type(v)}")
        return

    print(f"[inspect] Unsupported output type: {type(out)}")


def main():
    args = parse_args()
    cfg = load_config_and_patch(args)

    # Choose checkpoint
    if args.ckpt is not None:
        ckpt_path = args.ckpt
    else:
        # Use the latest ckpt under outputs/<job_id>/
        ckpt_path = get_latest_ckpt(cfg.log_dir)
        if ckpt_path is None:
            # Fallback: scan *.ckpt in the log_dir
            ckpts = glob.glob(os.path.join(cfg.log_dir, "*.ckpt"))
            ckpt_path = max(ckpts, key=os.path.getctime) if ckpts else None

    if ckpt_path is None:
        raise FileNotFoundError(
            f"No checkpoint provided and none found under {cfg.log_dir}. "
            f"Pass --ckpt explicitly."
        )

    print(f"[info] Using ckpt: {ckpt_path}")
    print(f"[info] Using dataset: {cfg.dataset_path} ({cfg.data_name}, basis={cfg.basis})")
    print(f"[info] Inspecting first {args.inspect_limit} prediction items (no files will be saved)")

    # Seed & data/model init
    pl.seed_everything(args.seed, workers=True)
    data = DataModule(cfg)
    model = LNNP(cfg)

    # Build a bare Trainer for prediction
    accelerator = "gpu" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        devices=1,
        accelerator=accelerator,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        precision=cfg.precision,
    )

    # Run predict
    preds = trainer.predict(model, datamodule=data, ckpt_path=ckpt_path)

    # preds is typically a list per batch; flatten lightweight for printing
    printed = 0
    for batch_out in preds:
        items = batch_out if isinstance(batch_out, (list, tuple)) else [batch_out]
        for o in items:
            if printed >= args.inspect_limit:
                break
            debug_print_output(printed, o, print_values=args.print_values)
            printed += 1
        if printed >= args.inspect_limit:
            break

    if printed == 0:
        print("[warn] No prediction items produced by trainer.predict().")


if __name__ == "__main__":
    main()

