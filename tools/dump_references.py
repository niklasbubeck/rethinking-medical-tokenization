#!/usr/bin/env python3
"""Write the real MedMNIST+ test images as the shared FID reference set.

calculate_fids.py (both modes), table_dataset_variance.py and
figS_reconstruction_matrix.py read the real test split from

  <base_dir>/outputs/<project>/reference/<dataset>/reference/<8-digit idx>.png

This dumps it once, in test-split order and with the same loader and PNG writer
that evaluate_generation.py uses for its samples (uint8 RGB, 224 x 224).

  python tools/dump_references.py
  python tools/dump_references.py --datasets breastmnist retinamnist
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate_generation import build_test_loader, save_samples_batch  # noqa: E402
from src import paths  # noqa: E402
from src.datasets.medmnist import two_d_datasets  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", nargs="+", default=list(two_d_datasets))
    ap.add_argument("--base_dir", default=paths.BASE_DIR)
    ap.add_argument("--project", default=paths.PROJECT)
    ap.add_argument("--data_root", default=paths.MEDMNIST_ROOT)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    a = ap.parse_args()

    for ds in a.datasets:
        out = os.path.join(a.base_dir, "outputs", a.project, "reference", ds, "reference")
        os.makedirs(out, exist_ok=True)
        _, loader = build_test_loader(ds, data_root=a.data_root, image_size=a.image_size,
                                      batch_size=a.bs, num_workers=a.num_workers)
        idx = 0
        for batch in loader:
            idx = save_samples_batch(batch["image"], out, idx)
        print(f"{ds}: {idx} images -> {out}")


if __name__ == "__main__":
    main()
