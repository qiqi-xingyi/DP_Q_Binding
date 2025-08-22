# --*-- conding:utf-8 --*--
# @time:8/21/25 17:22
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:predict.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any

import torch
import numpy as np
import pytorch_lightning as pl
from omegaconf import OmegaConf

# Add project root/src to PYTHONPATH
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.training.module import LNNP  # noqa: E402
from src.training.data import DataModule  # noqa: E402


# ----------------------------
# Helpers
# ----------------------------
def _infer_job_id_from_ckpt(ckpt_path: str) -> str:
    """Infer job_id from a checkpoint path. E.g., outputs/<job_id>/<file>.ckpt -> <job_id>."""
    p = Path(ckpt_path).resolve()
    # ckpt is outputs/<job_id>/<file>.ckpt
    return p.parent.name


def _load_cfg_from_job(job_id: str) -> Any:
    """Load the full training config saved by training at outputs/<job_id>/config.yaml."""
    cfg_path = Path(ROOT_DIR) / "outputs" / job_id / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Cannot find config file: {cfg_path}")
    return OmegaConf.load(str(cfg_path))


def _override_runtime_cfg(cfg: Any,
                          dataset_path: str = None,
                          data_name: str = None,
                          basis: str = None,
                          num_workers: int = None,
                          batch_size: int = None) -> Any:
    """Apply light runtime overrides without breaking the training config structure."""
    # Basic safety: disable training-time things that are irrelevant for inference
    cfg.ngpus = 1 if "ngpus" not in cfg or cfg.ngpus is None else cfg.ngpus
    cfg.num_sanity_val_steps = 0
    cfg.check_val_every_n_epoch = 1 if "check_val_every_n_epoch" not in cfg else cfg.check_val_every_n_epoch
    cfg.val_check_interval = None

    # Ensure deterministic small batch for inference unless overridden
    if batch_size is not None:
        cfg.inference_batch_size = batch_size
        cfg.batch_size = batch_size
    elif getattr(cfg, "inference_batch_size", None) is None:
        cfg.inference_batch_size = 1

    # Data related overrides
    if dataset_path is not None:
        cfg.dataset_path = dataset_path
    if data_name is not None:
        cfg.data_name = data_name
    if basis is not None:
        cfg.basis = basis
    if num_workers is not None:
        cfg.dataloader_num_workers = int(num_workers)

    # Logging dir remains the same job folder
    cfg.log_dir = os.path.join(cfg.log_dir, cfg.job_id)

    return cfg


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _print_batch_summary(i: int, batch) -> None:
    print(f"\n[inspect] Batch {i} keys: {list(batch.keys())}")
    keys_to_check = [
        "pred_hamiltonian_diagonal_blocks",
        "pred_hamiltonian_non_diagonal_blocks",
        "diag_mask",
        "non_diag_mask",
        "diag_hamiltonian",
        "non_diag_hamiltonian",
        "hamiltonian",         # GT full H (list) if exists
        "pred_hamiltonian"     # Pred full H (list) after build
    ]
    for k in keys_to_check:
        if hasattr(batch, k):
            v = getattr(batch, k)
            if isinstance(v, torch.Tensor):
                print(f"  {k}: tensor shape={tuple(v.shape)} dtype={v.dtype} device={v.device}")
            elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], torch.Tensor):
                print(f"  {k}: list(len={len(v)}) first shape={tuple(v[0].shape)}")
            else:
                print(f"  {k}: type={type(v)}")


def _save_per_molecule_matrices(save_dir: Path,
                                prefix: str,
                                matrices: List[torch.Tensor]) -> None:
    """Save a list of per-molecule square matrices as <prefix>_<i>.npy."""
    for i, mat in enumerate(matrices):
        npy_path = save_dir / f"{prefix}_{i}.npy"
        np.save(str(npy_path), _to_numpy(mat))


# ----------------------------
# Main
# ----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SPHNet prediction script (assemble full Hamiltonian and save).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to a checkpoint. If not provided, will use the latest ckpt in outputs/<job_id>/")
    parser.add_argument("--job_id", type=str, default=None,
                        help="Job id used during training. Required if --ckpt is not given.")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Override dataset path for inference.")
    parser.add_argument("--data_name", type=str, default=None,
                        help="Override data name (e.g., custom, qh9_stable_iid).")
    parser.add_argument("--basis", type=str, default=None,
                        help="Override basis (e.g., def2-svp).")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of workers for the test dataloader.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for inference.")
    parser.add_argument("--inspect_limit", type=int, default=0,
                        help="If > 0, only print summaries of the first N batches and exit (no files saved).")
    parser.add_argument("--save_root", type=str, default=None,
                        help="Override output directory to save predictions. Default: outputs/<job_id>/predictions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve job_id and ckpt
    if args.ckpt is None and args.job_id is None:
        raise ValueError("Either --ckpt or --job_id must be provided.")
    if args.job_id is None and args.ckpt is not None:
        args.job_id = _infer_job_id_from_ckpt(args.ckpt)

    # Load training config that was saved by train.py
    cfg = _load_cfg_from_job(args.job_id)

    # Merge runtime overrides
    cfg = _override_runtime_cfg(
        cfg,
        dataset_path=args.dataset_path,
        data_name=args.data_name,
        basis=args.basis,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    # Find ckpt if not specified
    if args.ckpt is None:
        job_dir = Path(ROOT_DIR) / "outputs" / args.job_id
        ckpts = sorted(job_dir.glob("*.ckpt"), key=os.path.getmtime)
        if not ckpts:
            raise FileNotFoundError(f"No .ckpt files found in {job_dir}")
        args.ckpt = str(ckpts[-1])

    # Where to save predictions
    save_root = Path(args.save_root) if args.save_root else (Path(ROOT_DIR) / "outputs" / args.job_id / "predictions")
    if args.inspect_limit <= 0:
        _ensure_dir(save_root)

    print(f"[info] Using ckpt: {args.ckpt}")
    print(f"[info] Using dataset: {cfg.dataset_path} ({cfg.data_name}, basis={cfg.basis})")
    if args.inspect_limit > 0:
        print(f"[info] Inspecting first {args.inspect_limit} batches (no files will be saved)")
    else:
        print(f"[info] Predictions will be saved to: {save_root}")

    # Seed and precision settings
    pl.seed_everything(int(cfg.seed) if "seed" in cfg else 123, workers=True)
    if str(cfg.precision) == "32":
        torch.set_float32_matmul_precision("high")

    # DataModule (use test split)
    dm = DataModule(cfg)
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LNNP.load_from_checkpoint(args.ckpt, hparams=cfg, map_location=device)
    model.eval().to(device)

    # Inference loop
    with torch.no_grad():
        for b_idx, batch in enumerate(test_loader):
            if args.inspect_limit > 0 and b_idx >= args.inspect_limit:
                break

            batch = batch.to(device)
            # Forward pass: predictions are written back into `batch`
            batch = model(batch)

            # Assemble full Hamiltonian matrices into `batch['pred_hamiltonian']`
            # This populates a per-molecule list of square tensors.
            model.hami_model.build_final_matrix(batch)

            if args.inspect_limit > 0:
                _print_batch_summary(b_idx, batch)
                continue

            # Save predictions
            # `pred_hamiltonian` is expected to be a list of per-molecule tensors
            if hasattr(batch, "pred_hamiltonian"):
                pred_list = batch["pred_hamiltonian"]
                _save_per_molecule_matrices(save_root, f"pred_H_batch{b_idx}", pred_list)
            else:
                # Fallback: save block outputs if full matrix is not present (shouldn't happen if build_final_matrix ran)
                for k in ["pred_hamiltonian_diagonal_blocks", "pred_hamiltonian_non_diagonal_blocks"]:
                    if hasattr(batch, k):
                        np.save(str(save_root / f"{k}_batch{b_idx}.npy"), _to_numpy(getattr(batch, k)))

            # Optionally save ground-truth matrices if they exist in the dataset
            if hasattr(batch, "hamiltonian"):
                gt_list = batch["hamiltonian"]
                _save_per_molecule_matrices(save_root, f"gt_H_batch{b_idx}", gt_list)

    if args.inspect_limit > 0:
        print("\n[done] Inspect mode finished.")
    else:
        print("\n[done] Predictions saved.")


if __name__ == "__main__":
    main()




