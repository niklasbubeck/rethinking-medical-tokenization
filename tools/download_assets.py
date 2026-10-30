#!/usr/bin/env python3
"""Download the data and pretrained assets used by the pipeline.

  medmnist  MedMNIST+ 2D datasets at 224x224 -> $MEDMNIST_ROOT/<dataset>_224.npz
  weights   MedMNIST v2 baseline ResNets (FMD feature extractor), Zenodo record
            7782114 (CC BY 4.0) -> $MEDMNIST_WEIGHTS/<dataset>/resnet18_224_1.pth
            Each archive is 0.9-1.6 GB; only the ResNet checkpoints are kept.
  sscd      SSCD copy-detection model (memorization) -> $SSCD_PATH
  dinov2    DINOv2 ViT-B/14 via torch.hub (memorization) -> torch hub cache

  python tools/download_assets.py                    # everything
  python tools/download_assets.py medmnist sscd      # a subset
  python tools/download_assets.py weights --datasets pathmnist octmnist
"""
import argparse
import os
import shutil
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import paths  # noqa: E402

DATASETS = ["pathmnist", "chestmnist", "dermamnist", "octmnist", "pneumoniamnist",
            "retinamnist", "breastmnist", "bloodmnist", "tissuemnist",
            "organamnist", "organcmnist", "organsmnist"]
ZENODO = "https://zenodo.org/api/records/7782114/files/weights_{ds}.zip/content"
SSCD_URL = "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"


def fetch(url, dst):
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    tmp = dst + ".part"
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(tmp, dst)


def get_medmnist(datasets):
    import medmnist
    from medmnist import INFO
    os.makedirs(paths.MEDMNIST_ROOT, exist_ok=True)
    for ds in datasets:
        cls = getattr(medmnist, INFO[ds]["python_class"])
        cls(split="train", download=True, size=224, root=paths.MEDMNIST_ROOT)
        print(f"  {ds}: {os.path.join(paths.MEDMNIST_ROOT, f'{ds}_224.npz')}")


def get_weights(datasets, keep):
    for ds in datasets:
        out = os.path.join(paths.MEDMNIST_WEIGHTS, ds)
        if os.path.isfile(os.path.join(out, "resnet18_224_1.pth")):
            print(f"  {ds}: already present")
            continue
        os.makedirs(out, exist_ok=True)
        archive = os.path.join(paths.MEDMNIST_WEIGHTS, f"weights_{ds}.zip")
        fetch(ZENODO.format(ds=ds), archive)
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                base = os.path.basename(name)
                if base.endswith(".pth") and base.startswith(keep):
                    with z.open(name) as src, open(os.path.join(out, base), "wb") as dst:
                        shutil.copyfileobj(src, dst)
        os.remove(archive)
        print(f"  {ds}: {sorted(os.listdir(out))}")


def get_sscd():
    if os.path.isfile(paths.SSCD_PATH):
        print(f"  already present: {paths.SSCD_PATH}")
        return
    fetch(SSCD_URL, paths.SSCD_PATH)


def get_dinov2():
    import torch
    if paths.TORCH_HUB_DIR:
        torch.hub.set_dir(paths.TORCH_HUB_DIR)
    torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", pretrained=True, verbose=False)
    print(f"  cached in {torch.hub.get_dir()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("assets", nargs="*", default=["medmnist", "weights", "sscd", "dinov2"],
                    choices=["medmnist", "weights", "sscd", "dinov2"])
    ap.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
    ap.add_argument("--keep", default="resnet18_224",
                    help="prefix of the checkpoints kept from each archive (default: the FMD backbone)")
    a = ap.parse_args()
    for asset in a.assets:
        print(f"[{asset}]")
        {"medmnist": lambda: get_medmnist(a.datasets),
         "weights": lambda: get_weights(a.datasets, a.keep),
         "sscd": get_sscd, "dinov2": get_dinov2}[asset]()


if __name__ == "__main__":
    main()
