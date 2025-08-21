# --*-- conding:utf-8 --*--
# @time:8/21/25 17:22
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:predict.py

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


def load_project_config(args):
    """
    Load config/config.yaml as-is (Hydra-style), make it non-struct, then override needed fields.
    Do NOT merge with a structured schema to avoid 'defaults' key conflicts.
    """
    default_cfg_path = os.path.join(repo_root, "config", "config.yaml")
    if not os.path.isfile(default_cfg_path):
        raise FileNotFoundError(f"Cannot find default config at {default_cfg_path}")

    cfg = OmegaConf.load(default_cfg_path)
    OmegaConf.set_struct(cfg, False)  # allow adding/overriding arbitrary keys

    # minimal runtime overrides
    cfg.job_id = args.job_id
    cfg.log_dir = os.path.join("outputs", cfg.job_id)
    cfg.data_name = args.data_name
    cfg.basis = args.basis
    cfg.dataset_path = args.dataset_path

    # inference knobs
    if getattr(cfg, "inference_batch_size", None) is None:
        cfg.inference_batch_size = 1
    cfg.dataloader_num_workers = args.num_workers
    cfg.ngpus = 1
    cfg.num_nodes = 1
    if not hasattr(cfg, "precision"):
        cfg.precision = "32"
    cfg.num_sanity_val_steps = 0
    cfg.check_val_every_n_epoch = 1

    # turn off wandb for inspection
    if "wandb" not in cfg:
        cfg.wandb = {}
    cfg.wandb["open"] = False

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
            print("  (values suppressed; set --print_values for small tensors only)")


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

    # resolve checkpoint path
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

    # datamodule & model
    data = DataModule(cfg)
    model = LNNP(cfg)

    # bare trainer for prediction
    accelerator = "gpu" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    trainer = pl.Trainer(
        devices=1,
        accelerator=accelerator,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        precision=cfg.precision,
    )

    # run predict
    preds = trainer.predict(model, datamodule=data, ckpt_path=ckpt_path)

    # print first N items
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


