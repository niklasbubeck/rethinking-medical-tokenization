"""
Codebook–latent misalignment entry script (reproduces Fig. 3 of DiVeQ).

Usage:
    python codebook_misalignment.py \
        --config configs/codebook_misalignment.yaml \
        --models discrete.vq.f16_d8_e16384 discrete.simvq.f16_d8_e16384 \
                 discrete.rqvae.f16_d8_e16384 discrete.lfq.f16_d12_b12 \
        [--datasets pneumoniamnist organamnist] \
        [--bs 128 --num_workers 8 --ckpt 1 --split test] \
        [--max_latents 5000 --max_codebook 4096 --perplexity 30] \
        [override.key=value ...]

For each (model, dataset) cell we:
  1. encode the test split, collect pre-quant latents P_z and post-quant z_q
  2. pull the learned codebook C_z from the quantizer
  3. project P_z ∪ C_z jointly with t-SNE and plot (gray points + red crosses)
  4. report distortion-per-bit (mean ||z - z_q||^2 / bits per token)
"""
import argparse
import time
import torch
from omegaconf import OmegaConf

from src.probing.misalignment import run_misalignment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/codebook_misalignment.yaml")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="Discrete encoder names in medlat (e.g. discrete.vq.f16_d8_e16384).")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Subset of MedMNIST 2D datasets. Default: all 12.")
    parser.add_argument("--ckpt", type=int, default=None,
                        help="Checkpoint step to load (default: latest).")
    parser.add_argument("--bs", type=int, default=None, help="Encoder forward batch size.")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val", "test"])
    parser.add_argument("--max_latents", type=int, default=None,
                        help="Cap on plotted latent vectors per cell.")
    parser.add_argument("--max_codebook", type=int, default=None,
                        help="Cap on plotted codebook entries per cell.")
    parser.add_argument("--perplexity", type=float, default=None,
                        help="t-SNE perplexity.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pool_datasets", dest="pool_datasets",
                        action="store_true", default=None,
                        help="One t-SNE per model over the union of all datasets (default).")
    parser.add_argument("--per_dataset", dest="pool_datasets",
                        action="store_false",
                        help="One t-SNE per (model, dataset).")
    parser.add_argument("--no_color_by_dataset", dest="color_by_dataset",
                        action="store_false", default=None,
                        help="In pooled mode, use a single color for all latent points.")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Override the run_name (default: auto timestamp).")
    parser.add_argument("--device", type=str, default=None)
    args, unknown = parser.parse_known_args()

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    cfg = OmegaConf.load(args.config)

    cfg.models = list(args.models)
    if args.datasets is not None:
        cfg.datasets = list(args.datasets)
    if args.ckpt is not None:
        cfg.ckpt = args.ckpt
    if args.bs is not None:
        cfg.batch_size = args.bs
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.split is not None:
        cfg.split = args.split
    if args.max_latents is not None:
        cfg.max_latents = args.max_latents
    if args.max_codebook is not None:
        cfg.max_codebook = args.max_codebook
    if args.perplexity is not None:
        cfg.perplexity = args.perplexity
    if args.seed is not None:
        cfg.seed = args.seed
    if args.pool_datasets is not None:
        cfg.pool_datasets = args.pool_datasets
    if args.color_by_dataset is not None:
        cfg.color_by_dataset = args.color_by_dataset
    if args.device is not None:
        cfg.device = args.device
    if args.exp_name:
        cfg.run_name = args.exp_name

    if unknown:
        unknown = [a.lstrip("-") if a.startswith("-") else a for a in unknown]
        cli_conf = OmegaConf.from_cli(unknown)
        cfg = OmegaConf.merge(cfg, cli_conf)

    if not cfg.run_name:
        cfg.run_name = time.strftime("misalign_%Y%m%d_%H%M%S")

    print(OmegaConf.to_yaml(cfg))
    run_misalignment(OmegaConf.to_container(cfg, resolve=True))


if __name__ == "__main__":
    main()
