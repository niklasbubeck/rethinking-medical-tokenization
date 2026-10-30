"""
evaluate_generation.py — generate samples from each trained DiT and dump them
to disk as PNGs.

Standalone, single-script counterpart to evaluate_tokenizer.py. For every
(model, dataset) combination this script:

  1. Loads the frozen tokenizer and the trained DiT generator (wrapped in
     GenWrapper, matching the training pipeline).
  2. Walks the test split of the dataset, taking (dataset_id, class_label)
     from each batch to drive class- and dataset-conditioned generation.
  3. Runs the diffusion sampler on Gaussian noise of the correct latent
     shape, decodes through the frozen tokenizer, and writes each sample
     as a flat-numbered PNG under

       <base_dir>/outputs/<project>/dit_<model>/generation/<dataset>/<00000000>.png

FID, MedFID, self-MS-SSIM and any other distribution-level metric is NOT
computed here. Compute them offline from the saved PNGs. Keeping the GPU
continuously busy on sampling avoids the 90-minute idle-watchdog that was
killing long DiT-test jobs on the universe partition.
"""

import argparse
import json
import os

import time

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.io import write_png
from tqdm import tqdm

from src import paths
from src.datasets.medmnist import MergedMedMNIST, two_d_datasets
from src.diffusion import create_gaussian_diffusion
from src.utils import (
    find_latest_checkpoint,
    load_checkpoint,
    remove_module_prefix,
    seed_worker,
    seed_everything,
    training_reproducibility_cudnn,
)
from medlat import get_model, GenWrapper


# ============================================================================
# Diffusion defaults — match what the DiT training run used. The only knob the
# user is expected to vary is `timestep_respacing` (sampling speed/quality
# trade-off) and `cfg_scale` (classifier-free guidance strength).
# ============================================================================
DIFFUSION_DEFAULTS = dict(
    steps=1000,
    learn_sigma=True,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
)


# ============================================================================
def load_dit_model(model_name, *, base_dir, project, ckpt, img_size, device):
    """Reconstruct the GenWrapper used during training and load the DiT ckpt.

    Mirrors the load sequence in the original DiT test loop:
      - The tokenizer ckpt lives at <base_dir>/outputs/<project>/<model>/.
      - The DiT ckpt lives at <base_dir>/outputs/<project>/dit_<model>/.
      - The DiT state dict contains weights for BOTH submodels, so loading it
        into GenWrapper is enough.
    """
    tokenizer_dir = os.path.join(base_dir, "outputs", project, model_name)
    dit_dir = os.path.join(base_dir, "outputs", project, f"dit_{model_name}")

    tok_ckpt = find_latest_checkpoint(tokenizer_dir)
    print(f"[gen] tokenizer ckpt: {tok_ckpt}")
    tokenizer = get_model(model_name, img_size=img_size, ckpt_path=tok_ckpt)

    generator = get_model(
        "dit.b_2",
        img_size=img_size,
        vae_stride=tokenizer.vae_stride,
        in_channels=tokenizer.embed_dim,
        dataset_num=12,
        dataset_label_drop_prob=0.0,
    )
    model = GenWrapper(generator, tokenizer)

    if ckpt is not None:
        dit_ckpt = os.path.join(dit_dir, "train", "models", f"ckpt.{ckpt}.pt")
    else:
        dit_ckpt = find_latest_checkpoint(dit_dir)
    print(f"[gen] dit ckpt:       {dit_ckpt}")

    state = load_checkpoint(dit_ckpt)
    # strict=False: the training pipeline added auto-scale-factor buffers
    # (_running_std_sum, _running_std_count) after some DiTs were checkpointed.
    # They are training-only (early-return in eval) and never read at inference,
    # so loading without them is safe; the persisted scale_factor itself is in
    # the checkpoint and IS loaded.
    missing, unexpected = model.load_state_dict(
        remove_module_prefix(state["model"]), strict=False)
    if missing:
        print(f"[gen] load_state_dict missing keys: {missing}")
    if unexpected:
        print(f"[gen] load_state_dict unexpected keys: {unexpected}")
    model.to(device).eval()
    return model, dit_dir, dit_ckpt


def build_test_loader(dataset_name, *, data_root, image_size, batch_size, num_workers):
    """Single-dataset MedMNIST test loader. shuffle=False so the i-th saved PNG
    corresponds to the i-th sample of the test split (useful for class lookup
    in post-processing)."""
    ds = MergedMedMNIST(
        root=data_root,
        split="test",
        transform=T.ToTensor(),
        ds_names=[dataset_name],
        size=image_size,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
        worker_init_fn=seed_worker,
    )
    return ds, loader


# ============================================================================
def save_samples_batch(samples, out_dir, start_idx, *, rescale_from_neg11=False):
    """Write a batch of images as 8-digit-numbered PNGs into out_dir.

    Default assumes input is already in [0, 1] (matches the original
    ``ToDisk.vid_or_image_to_png`` and ``trainer.normalize: False`` in the
    configs/dit.yaml). Set ``rescale_from_neg11=True`` only for models trained
    with ``trainer.normalize: True`` (decoder output in [-1, 1]); otherwise the
    rescale would just shift everything brighter and wash out colors.

    Images are clamped, forced to 3-channel RGB (Inception / MedMNIST
    classifiers expect RGB), and quantized to uint8. Returns the next free index.
    """
    x = samples.detach().float()
    if rescale_from_neg11:
        x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)
    if x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1)
    elif x.shape[1] > 3:
        x = x[:, :3]
    x_u8 = (x * 255).round().to(torch.uint8).cpu()
    n = x_u8.shape[0]
    for i in range(n):
        write_png(x_u8[i], os.path.join(out_dir, f"{start_idx + i:08d}.png"))
    return start_idx + n


# ============================================================================
def generate_for_dataset(model, loader, *, sampler_fcn, cfg_scale, cfg_dataset,
                         temperature, clip_denoised, dataset_name,
                         sample_out_dir, reference_out_dir,
                         sample_rescale_from_neg11, save_references,
                         device, autocast_dtype=torch.bfloat16):
    """For every test image: sample a class-conditioned generation, decode,
    write to ``sample_out_dir``. When ``save_references`` is True, the matching
    real test image is also written to ``reference_out_dir`` under the same
    8-digit index. Off by default to avoid writing the identical MedMNIST
    reference set N times (once per tokenizer); enable for the first run if
    you don't already have a canonical reference dump.

    The GT image is encoded only to learn the latent shape; sampling itself
    starts from fresh Gaussian noise, conditioned on (label, dataset_id) read
    from the batch."""
    os.makedirs(sample_out_dir, exist_ok=True)
    if save_references:
        os.makedirs(reference_out_dir, exist_ok=True)
    image_idx = 0
    t0 = time.time()

    for batch in tqdm(loader, leave=False, dynamic_ncols=True, desc=dataset_name):
        img = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].squeeze(1).to(device, non_blocking=True)
        dataset_ids = batch["dataset_id"].to(device, non_blocking=True)

        with torch.no_grad():
            # encode GT only to grab the latent shape; we throw the latent away
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                z_e = model.vae_encode(img)
            shape = z_e.shape

            cond = dict(
                y=labels,
                dataset_id=dataset_ids,
                cfg_scale=cfg_scale,
                cfg_dataset=cfg_dataset,
            )

            sample_latent = sampler_fcn(
                model.generator.forward_with_cfg,
                shape,
                clip_denoised=clip_denoised,
                progress=False,
                model_kwargs=cond,
                temperature=temperature,
                device=device,
            )

            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                sample = model.vae_decode(sample_latent)

        # Sample range depends on the tokenizer: trainer.normalize=False → [0, 1]
        # (default in configs/dit.yaml), trainer.normalize=True → [-1, 1].
        # Reference (T.ToTensor()) is always [0, 1].
        next_idx = save_samples_batch(sample, sample_out_dir, image_idx,
                                      rescale_from_neg11=sample_rescale_from_neg11)
        if save_references:
            save_samples_batch(img, reference_out_dir, image_idx,
                               rescale_from_neg11=False)
        image_idx = next_idx

    elapsed = time.time() - t0
    return {
        "n_samples": image_idx,
        "elapsed_sec": round(elapsed, 2),
        "throughput_img_s": round(image_idx / max(elapsed, 1e-6), 2),
        "sample_dir": sample_out_dir,
        "reference_dir": reference_out_dir if save_references else None,
    }


# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    # data + model selection
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="tokenizer names; the DiT lives at "
                             "dit_<model>/. Multiple allowed.")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Subset of MedMNIST 2D datasets; default = all 12.")
    parser.add_argument("--base_dir", type=str, default=paths.BASE_DIR,
                        help="Folder containing outputs/<project>/ ($RMT_BASE_DIR).")
    parser.add_argument("--project", type=str, default=paths.PROJECT)
    parser.add_argument("--data_root", type=str, default=paths.MEDMNIST_ROOT,
                        help="MedMNIST+ .npz folder ($MEDMNIST_ROOT).")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ckpt", type=int, default=None,
                        help="DiT checkpoint step; default = latest.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # diffusion / sampler controls
    parser.add_argument("--sampler", type=str, default="ddpm",
                        choices=["ddpm", "ddim"])
    parser.add_argument("--timestep_respacing", type=str, default="100",
                        help="Passed to create_gaussian_diffusion.")
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--cfg_dataset", action="store_true",
                        help="If set, also apply CFG to the dataset embedding.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--clip_denoised", action="store_true")
    parser.add_argument("--normalize", action="store_true",
                        help="Rescale generated samples from [-1, 1] to [0, 1] "
                             "before saving. Off by default — matches "
                             "trainer.normalize=False in configs/dit.yaml. "
                             "Only enable for tokenizers that were trained "
                             "with trainer.normalize=True.")
    parser.add_argument("--save_references", action="store_true",
                        help="Also dump the real MedMNIST test images to a "
                             "sibling reference/ folder per (model, dataset). "
                             "Off by default to avoid writing the identical "
                             "reference set N times (once per tokenizer). "
                             "Enable for the first run to populate a "
                             "canonical reference dump.")

    # output
    parser.add_argument("--output_subdir", type=str, default="generation",
                        help="Subdir under dit_<model>/ where PNGs land.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run a (model, dataset) cell even if its "
                             "output folder already contains PNGs.")

    args = parser.parse_args()

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    seed_everything(args.seed)
    training_reproducibility_cudnn()

    device = torch.device(args.device)
    datasets = list(args.datasets) if args.datasets else list(two_d_datasets)

    diffusion = create_gaussian_diffusion(
        **DIFFUSION_DEFAULTS,
        timestep_respacing=args.timestep_respacing,
    )
    sampler_fcn = (diffusion.p_sample_loop if args.sampler == "ddpm"
                   else diffusion.ddim_sample_loop)

    print(f"[gen] models   ({len(args.models)}): {args.models}")
    print(f"[gen] datasets ({len(datasets)}): {datasets}")
    print(f"[gen] sampler={args.sampler}  steps={args.timestep_respacing}  "
          f"cfg={args.cfg_scale}  temp={args.temperature}  "
          f"clip={args.clip_denoised}")

    for model_name in args.models:
        print(f"\n========== MODEL: {model_name} ==========")
        try:
            model, dit_dir, dit_ckpt = load_dit_model(
                model_name,
                base_dir=args.base_dir, project=args.project,
                ckpt=args.ckpt, img_size=args.image_size, device=device,
            )
        except Exception as e:
            print(f"[gen] FAIL load {model_name}: {e}")
            import traceback; traceback.print_exc()
            continue

        gen_root = os.path.join(dit_dir, args.output_subdir)
        os.makedirs(gen_root, exist_ok=True)
        print(f"[gen] writing samples under: {gen_root}")

        per_model_stats = {}
        model_t0 = time.time()
        for ds_name in datasets:
            ds_root = os.path.join(gen_root, ds_name)
            sample_out_dir = os.path.join(ds_root, "sample")
            reference_out_dir = os.path.join(ds_root, "reference")

            if (not args.overwrite) and os.path.isdir(sample_out_dir) \
                    and any(f.endswith(".png") for f in os.listdir(sample_out_dir)):
                print(f"--- {model_name} :: {ds_name}: SKIP "
                      f"(sample/ already has PNGs; pass --overwrite to redo)")
                continue

            print(f"\n--- {model_name} :: {ds_name} ---")
            try:
                _, loader = build_test_loader(
                    ds_name, data_root=args.data_root,
                    image_size=args.image_size,
                    batch_size=args.bs, num_workers=args.num_workers,
                )
                stats = generate_for_dataset(
                    model, loader,
                    sampler_fcn=sampler_fcn,
                    cfg_scale=args.cfg_scale,
                    cfg_dataset=args.cfg_dataset,
                    temperature=args.temperature,
                    clip_denoised=args.clip_denoised,
                    dataset_name=ds_name,
                    sample_out_dir=sample_out_dir,
                    reference_out_dir=reference_out_dir,
                    sample_rescale_from_neg11=args.normalize,
                    save_references=args.save_references,
                    device=device,
                )
            except Exception as e:
                print(f"[gen] FAIL {model_name}::{ds_name}: {e}")
                import traceback; traceback.print_exc()
                continue
            kind = "sample/+reference/" if args.save_references else "sample/"
            print(f"  saved {stats['n_samples']} {kind} PNGs "
                  f"under {ds_root} in {stats['elapsed_sec']:.1f}s "
                  f"({stats['throughput_img_s']:.1f} img/s)")
            per_model_stats[ds_name] = stats

        # Per-model run summary written next to the samples, for reproducibility.
        manifest = {
            "model": model_name,
            "dit_ckpt": dit_ckpt,
            "sampler": args.sampler,
            "timestep_respacing": args.timestep_respacing,
            "cfg_scale": args.cfg_scale,
            "cfg_dataset": args.cfg_dataset,
            "temperature": args.temperature,
            "clip_denoised": args.clip_denoised,
            "image_size": args.image_size,
            "bs": args.bs,
            "seed": args.seed,
            "device": str(device),
            "model_elapsed_sec": round(time.time() - model_t0, 2),
            "datasets": per_model_stats,
        }
        with open(os.path.join(gen_root, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n[gen] {model_name} done in {manifest['model_elapsed_sec']:.1f}s "
              f"(manifest at {os.path.join(gen_root, 'manifest.json')})")


if __name__ == "__main__":
    main()
