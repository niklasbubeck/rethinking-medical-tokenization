#!/usr/bin/env python3
"""
Offline FID and FMD over already-dumped sample vs. reference PNGs, for both
reconstruction (reconstructions vs. real test images) and generation (DiT
samples vs. real test images). Computes:

  * FID: standard Inception-V3 Frechet distance (dataset-agnostic backbone).
  * FMD (called MedFID in the code and CSV logs): Frechet distance in the feature
    space of the official MedMNIST baseline ResNet (224 variant). The feature
    extractor is dataset-specific: each dataset uses its own classifier
    checkpoint, <weights_root>/<dataset>/resnet{18,50}_224_<seed>.pth
    (download with tools/download_assets.py weights).

Reuses the PNG directories written by evaluate_tokenizer.py and
evaluate_generation.py, so no model re-encoding is needed. Rows are appended to
--out_csv; analysis/aggregate_results.py keeps the latest row per configuration.

Directory layout (<base_dir>/outputs/<project>/):
  reference/<ds>/reference/*.png                # real test split (tools/dump_references.py)
  dit_<model>/generation/<ds>/sample/*.png      # generation samples
  <model>/reconstruction/<ds>/sample/*.png      # reconstructions

Examples:
  python calculate_fids.py --mode generation --metrics both \
      --models discrete.vq.f8_d4_e16384 discrete.rqvae.f8_d4_e16384 \
      --out_csv fids_generation.csv

  # FMD only, with the ResNet-50 backbone:
  python calculate_fids.py --mode reconstruction --metrics fmd --arch resnet50 \
      --models discrete.vq.f8_d4_e16384 --out_csv fids_recon_r50.csv
"""
import argparse
import csv
import glob
import os

import torch

import torch_fidelity

from src import paths

ALL_2D = [
    "pathmnist", "chestmnist", "dermamnist", "octmnist", "pneumoniamnist",
    "retinamnist", "breastmnist", "bloodmnist", "tissuemnist",
    "organamnist", "organcmnist", "organsmnist",
]
INCEPTION = "inception-v3-compat"


def n_pngs(d):
    return len(glob.glob(os.path.join(d, "*.png"))) if d and os.path.isdir(d) else 0


def resolve_dirs(base_proj, model, dataset, mode):
    """Return (sample_dir, reference_dir) for a (model, dataset, mode)."""
    if mode == "generation":
        sample = os.path.join(base_proj, f"dit_{model}", "generation", dataset, "sample")
        reference = os.path.join(base_proj, "reference", dataset, "reference")
    elif mode == "reconstruction":
        sample = os.path.join(base_proj, model, "reconstruction", dataset, "sample")
        # use the shared real test-split dump as reference (recon was run without
        # --save_references); rFID = FID(reconstructions, real test images).
        reference = os.path.join(base_proj, "reference", dataset, "reference")
    else:
        raise ValueError(mode)
    return sample, reference


def frechet(sample_dir, reference_dir, fe_name, weights_path, cuda):
    """Single FID call with the given feature extractor; returns the float."""
    kwargs = dict(
        input1=sample_dir, input2=reference_dir,
        feature_extractor=fe_name,
        cuda=cuda,
        isc=False, fid=True, kid=False, prc=False,
        verbose=False, samples_find_deep=False,
    )
    if weights_path is not None:
        kwargs["feature_extractor_weights_path"] = weights_path
    out = torch_fidelity.calculate_metrics(**kwargs)
    return float(out.get("frechet_inception_distance", float("nan")))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True,
                   help="Canonical model names, e.g. discrete.vq.f8_d4_e16384")
    p.add_argument("--mode", choices=["generation", "reconstruction"], required=True)
    p.add_argument("--metrics", choices=["fid", "fmd", "both"], default="both")
    p.add_argument("--datasets", nargs="+", default=None, help="Subset; default all 12 2D datasets.")
    p.add_argument("--arch", choices=["resnet18", "resnet50"], default="resnet18",
                   help="MedMNIST backbone for FMD (Frechet MedMNIST Distance).")
    p.add_argument("--seed", type=int, default=1, help="MedMNIST checkpoint seed (1-3).")
    p.add_argument("--base_dir", default=paths.BASE_DIR,
                   help="Folder containing outputs/<project>/ ($RMT_BASE_DIR).")
    p.add_argument("--project", default=paths.PROJECT)
    p.add_argument("--weights_root", default=paths.MEDMNIST_WEIGHTS,
                   help="MedMNIST ResNet baselines, <dataset>/resnet18_224_1.pth ($MEDMNIST_WEIGHTS).")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    want_fid = args.metrics in ("fid", "both")
    want_fmd = args.metrics in ("fmd", "both")
    fmd_fe = f"medmnist-{args.arch}"
    datasets = args.datasets or ALL_2D
    base_proj = os.path.join(args.base_dir, "outputs", args.project)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    fields = ["model", "dataset", "mode", "fid", "fmd", "fmd_arch", "seed",
              "n_sample", "n_reference"]
    write_header = not os.path.exists(args.out_csv)
    fcsv = open(args.out_csv, "a", newline="")
    writer = csv.DictWriter(fcsv, fieldnames=fields)
    if write_header:
        writer.writeheader()

    for model in args.models:
        for ds in datasets:
            sample_dir, reference_dir = resolve_dirs(base_proj, model, ds, args.mode)
            ns, nr = n_pngs(sample_dir), n_pngs(reference_dir)
            tag = f"[{model} | {ds} | {args.mode}]"
            if ns == 0 or nr == 0:
                print(f"{tag} SKIP: sample={ns} reference={nr} ({sample_dir} | {reference_dir})", flush=True)
                continue

            fid_v = float("nan")
            fmd_v = float("nan")

            if want_fid:
                try:
                    fid_v = frechet(sample_dir, reference_dir, INCEPTION, None, cuda)
                except Exception as e:
                    print(f"{tag} FID ERROR: {type(e).__name__}: {e}", flush=True)

            if want_fmd:
                wpath = os.path.join(args.weights_root, ds, f"{args.arch}_224_{args.seed}.pth")
                if not os.path.isfile(wpath):
                    print(f"{tag} FMD SKIP: no weights {wpath}", flush=True)
                else:
                    try:
                        fmd_v = frechet(sample_dir, reference_dir, fmd_fe, wpath, cuda)
                    except Exception as e:
                        print(f"{tag} FMD ERROR: {type(e).__name__}: {e}", flush=True)

            row = {"model": model, "dataset": ds, "mode": args.mode,
                   "fid": fid_v, "fmd": fmd_v,
                   "fmd_arch": args.arch if want_fmd else "",
                   "seed": args.seed, "n_sample": ns, "n_reference": nr}
            writer.writerow(row)
            fcsv.flush()
            print(f"{tag} FID={fid_v:.4f}  FMD={fmd_v:.4f}  "
                  f"(n_sample={ns}, n_ref={nr})", flush=True)

    fcsv.close()
    print(f"\nWrote -> {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
