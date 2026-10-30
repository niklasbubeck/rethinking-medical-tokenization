"""
Linear probing entry script.

Usage:
    python linear_probing.py \
        --config configs/linear_probe.yaml \
        --models continuous.aekl.f4_d32 token.titok.b_128_p8_e2e \
        [--datasets pneumoniamnist organamnist] \
        [--bs 256 --num_workers 8 --ckpt 1] \
        [override.key=value ...]

Encoder names must be present in `medlat.available_models()` and have a trained
checkpoint under <base_dir>/outputs/<project>/<name>/train/models/ckpt.<step>.pt.
"""
import argparse
import time
import torch
from omegaconf import OmegaConf

from src.probing.runner import run_probing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/linear_probe.yaml")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="Encoder names registered in medlat (e.g. continuous.aekl.f4_d32).")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Subset of MedMNIST 2D datasets. Default: all 12.")
    parser.add_argument("--ckpt", type=int, default=None,
                        help="Checkpoint step to load (default: latest).")
    parser.add_argument("--bs", type=int, default=None, help="Encoder forward batch size.")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--pooling", type=str, default=None, choices=["flatten", "mean", "cls"],
                        help="Feature pooling before the linear head.")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Override the run_name (default: auto timestamp).")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--debug", action="store_true",
                        help="Reserved for parity with train.py; currently has no effect.")
    args, unknown = parser.parse_known_args()

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    cfg = OmegaConf.load(args.config)

    # Apply explicit CLI args
    cfg.models = list(args.models)
    if args.datasets is not None:
        cfg.datasets = list(args.datasets)
    if args.ckpt is not None:
        cfg.ckpt = args.ckpt
    if args.bs is not None:
        cfg.batch_size = args.bs
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.pooling is not None:
        cfg.pooling = args.pooling
    if args.device is not None:
        cfg.device = args.device
    if args.exp_name:
        cfg.run_name = args.exp_name

    # Apply free-form key=value overrides (OmegaConf style)
    if unknown:
        unknown = [a.lstrip("-") if a.startswith("-") else a for a in unknown]
        cli_conf = OmegaConf.from_cli(unknown)
        cfg = OmegaConf.merge(cfg, cli_conf)

    if not cfg.run_name:
        cfg.run_name = time.strftime("probe_%Y%m%d_%H%M%S")

    print(OmegaConf.to_yaml(cfg))
    run_probing(OmegaConf.to_container(cfg, resolve=True))


if __name__ == "__main__":
    main()
