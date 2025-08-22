# --*-- conding:utf-8 --*--
# @time:8/21/25 17:22
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:predict.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import argparse

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

# repo root
cur_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.join(cur_dir, "..")
sys.path.append(repo_root)

from src.training.data import DataModule
from src.training.module import LNNP
from src.training.logger import get_latest_ckpt

try:
    from torch_geometric.data import Data as TGData
except Exception:
    TGData = None


def parse_args():
    p = argparse.ArgumentParser(description="Inspect-only predictor: print model prediction fields (no saving).")
    p.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint. If omitted, use latest under outputs/<job_id>/")
    p.add_argument("--job_id", type=str, default="train_quick", help="Folder under outputs/ to read ckpt from if --ckpt not set")
    p.add_argument("--dataset_path", type=str, required=True, help="Path to dataset (e.g., ./example_data/data.mdb or .lmdb)")
    p.add_argument("--data_name", type=str, default="custom", help="Dataset split name (custom/qh9_stable_iid/...)")
    p.add_argument("--basis", type=str, default="def2-svp", help="Basis string (e.g., def2-svp)")
    p.add_argument("--num_workers", type=int, default=0, help="predict dataloader workers (0 avoids fork issues)")
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Accelerator hint")
    p.add_argument("--inspect_limit", type=int, default=3, help="How many prediction items to print")
    p.add_argument("--print_values", action="store_true", help="Also print values for small (<=10x10) tensors")
    p.add_argument("--seed", type=int, default=123, help="Random seed")
    return p.parse_args()


def ensure_runtime_defaults(cfg):
    OmegaConf.set_struct(cfg, False)
    # Heads/toggles the model code may access at predict time
    for k, v in {
        "enable_hami": True,
        "enable_energy": False,
        "enable_forces": False,
        "enable_symmetry": False,
        "enable_energy_hami_error": False,
        "enable_hami_orbital_energy": False,
    }.items():
        cfg.setdefault(k, v)
    for k, v in {
        "energy_weight": 0.0,
        "forces_weight": 0.0,
        "hami_weight": 1.0,
        "orbital_energy_weight": 0.0,
    }.items():
        cfg.setdefault(k, v)
    cfg.setdefault("hami_train_loss", "maemse")
    cfg.setdefault("hami_val_loss", "mae")
    cfg.setdefault("precision", "32")
    cfg.setdefault("num_sanity_val_steps", 0)
    cfg.setdefault("check_val_every_n_epoch", 1)
    if "wandb" not in cfg:
        cfg["wandb"] = {}
    cfg.wandb["open"] = False


def ensure_model_defaults(cfg):
    """Provide safe defaults for model & hami head if missing (values mirror your train logs)."""
    OmegaConf.set_struct(cfg, False)
    if "model" not in cfg or cfg.model is None:
        cfg.model = {
            "order": 4,
            "embedding_dimension": 128,
            "bottle_hidden_size": 32,
            "max_radius": 15,
            "num_nodes": 10,
            "radius_embed_dim": 16,
            "use_equi_norm": True,
            "short_cutoff_upper": 5,
            "long_cutoff_upper": 15,
            "num_scale_atom_layers": 4,
            "num_long_range_layers": 4,
        }
    if "hami_model" not in cfg or cfg.hami_model is None:
        cfg.hami_model = {
            "name": "HamiHead_sphnet",
            "irreps_edge_embedding": None,
            "num_layer": 2,
            "max_radius_cutoff": 30,
            "radius_embed_dim": 16,
            "bottle_hidden_size": 32,
        }
    # Sparse TP flags also showed up in your runs
    cfg.setdefault("use_sparse_tp", True)
    cfg.setdefault("sparsity", 0.7)


def load_project_config(args):
    """Load config/config.yaml, then override minimal fields and ensure defaults."""
    default_cfg_path = os.path.join(repo_root, "config", "config.yaml")
    if not os.path.isfile(default_cfg_path):
        raise FileNotFoundError(f"Cannot find default config at {default_cfg_path}")

    cfg = OmegaConf.load(default_cfg_path)
    OmegaConf.set_struct(cfg, False)

    # Minimal runtime overrides
    cfg.job_id = args.job_id
    cfg.log_dir = os.path.join("outputs", cfg.job_id)
    cfg.data_name = args.data_name
    cfg.basis = args.basis
    cfg.dataset_path = args.dataset_path
    if getattr(cfg, "inference_batch_size", None) is None:
        cfg.inference_batch_size = 1
    cfg.dataloader_num_workers = args.num_workers
    cfg.ngpus = 1
    cfg.num_nodes = 1

    ensure_runtime_defaults(cfg)
    ensure_model_defaults(cfg)
    return cfg


def pretty_print_tensor(name, t, print_values=False, max_print=10):
    t_cpu = t.detach().cpu()
    desc = f"- {name}: Tensor shape={tuple(t_cpu.shape)}, dtype={t_cpu.dtype}"
    try:
        desc += f", min={float(t_cpu.min()):.6g}, max={float(t_cpu.max()):.6g}"
    except Exception:
        pass
    print(desc)
    if print_values:
        numel = t_cpu.numel()
        if t_cpu.ndim <= 2 and t_cpu.shape[0] <= max_print and (t_cpu.ndim == 1 or t_cpu.shape[-1] <= max_print) and numel <= max_print * max_print:
            print(t_cpu)
        else:
            print("  (values suppressed; enable --print_values for small tensors only)")


def debug_print_output(idx, out, print_values=False):
    print(f"\n[inspect] >>> Item #{idx}")
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
    cfg = load_project_config(args)

    # Resolve checkpoint
    if args.ckpt is not None:
        ckpt_path = args.ckpt
    else:
        ckpt_path = get_latest_ckpt(cfg.log_dir)
        if ckpt_path is None:
            ckpts = glob.glob(os.path.join(cfg.log_dir, "*.ckpt"))
            ckpt_path = max(ckpts, key=os.path.getctime) if ckpts else None
    if ckpt_path is None:
        raise FileNotFoundError(f"No checkpoint found. Pass --ckpt or ensure *.ckpt exists under {cfg.log_dir}")

    print(f"[info] Using ckpt: {ckpt_path}")
    print(f"[info] Using dataset: {cfg.dataset_path} ({cfg.data_name}, basis={cfg.basis})")
    print(f"[info] Inspecting first {args.inspect_limit} items (no files will be saved)")

    pl.seed_everything(args.seed, workers=True)

    data = DataModule(cfg)
    model = LNNP(cfg)

    accelerator = "gpu" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    trainer = pl.Trainer(
        devices=1,
        accelerator=accelerator,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        precision=str(cfg.precision),
    )

    preds = trainer.predict(model, datamodule=data, ckpt_path=ckpt_path)

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



