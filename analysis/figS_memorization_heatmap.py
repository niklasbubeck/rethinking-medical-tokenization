#!/usr/bin/env python3
"""
Figure S8 - Memorization by tokenizer family x dataset (2x2 heatmap).

Rows:    Copy-rate (top)  and  NN-distance ratio (bottom)
Columns: SSCD (copy detection, left)  and  DINOv2 (semantic, right)
One (family x dataset) heatmap per cell, at a single compression factor (f=8).

Reads memorization.csv (model, dataset, extractor, copy_rate, mem_ratio).

  python analysis/figS_memorization_heatmap.py           # f=8 (paper)
  python analysis/figS_memorization_heatmap.py --f 16
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import _paths

INK = "#1a1714"

# row order (families) and their display labels
FAM_KEYS = ["aekl", "soft_vq", "vq", "bsq", "lfq", "diveq",
            "sf_diveq", "simple_qinco", "simvq", "rqvae"]
FAM_LABEL = {
    "aekl": "VAEKL", "soft_vq": "SoftVQ", "vq": "VQGAN", "bsq": "BSQ",
    "lfq": "LFQ", "diveq": "DiVeQ", "sf_diveq": "SF-DiVeQ",
    "simple_qinco": "SimQinco", "simvq": "SimVQ", "rqvae": "RQVAE",
}
# column order (datasets) and their display labels
DS_KEYS = ["pathmnist", "octmnist", "dermamnist", "bloodmnist", "tissuemnist",
           "organamnist", "organcmnist", "organsmnist", "pneumoniamnist",
           "retinamnist", "breastmnist", "chestmnist"]
DS_LABEL = {
    "pathmnist": "PathMNIST", "octmnist": "OctMNIST", "dermamnist": "DermaMNIST",
    "bloodmnist": "BloodMNIST", "tissuemnist": "TissueMNIST",
    "organamnist": "OrganAMNIST", "organcmnist": "OrganCMNIST",
    "organsmnist": "OrganSMNIST", "pneumoniamnist": "PneumoniaMNIST",
    "retinamnist": "RetinaMNIST", "breastmnist": "BreastMNIST",
    "chestmnist": "ChestMNIST",
}
EXTRACTORS = [("sscd", "SSCD (copy detection)"), ("dinov2", "DINOv2 (semantic)")]

# color / range settings per metric (match the paper S8)
COPY_CMAP, COPY_VMIN, COPY_VMAX = "YlOrBr", 0.0, 0.20
RATIO_CMAP, RATIO_VMIN, RATIO_VMAX = "Purples_r", 0.74, 0.86   # darker = tighter
COPY_ANNOT_MIN = 0.045     # annotate copy-rate cells at/above this
RATIO_ANNOT_MAX = 0.86     # annotate NN-ratio cells at/below this


def fam_of(model):
    return model.split(".")[1]


def comp_of(model):
    import re
    m = re.search(r"\.f(\d+)_", model)
    return int(m.group(1)) if m else None


def matrix(df, ext, col):
    """(family x dataset) value matrix for one extractor/metric; NaN if absent."""
    sub = df[df.extractor == ext]
    piv = sub.pivot_table(index="fam", columns="dataset", values=col, aggfunc="mean")
    M = np.full((len(FAM_KEYS), len(DS_KEYS)), np.nan)
    for i, f in enumerate(FAM_KEYS):
        for j, d in enumerate(DS_KEYS):
            if f in piv.index and d in piv.columns:
                M[i, j] = piv.loc[f, d]
    return M


def draw_heat(ax, M, cmap, vmin, vmax, annot_fn):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(DS_KEYS)))
    ax.set_yticks(range(len(FAM_KEYS)))
    ax.set_yticklabels([FAM_LABEL[f] for f in FAM_KEYS], fontsize=10, color=INK)
    ax.set_xticks(np.arange(-.5, len(DS_KEYS), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(FAM_KEYS), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)
    # per-cell annotations (value + contrast-aware text color)
    norm = plt.Normalize(vmin, vmax)
    cmap_obj = plt.get_cmap(cmap)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v) or not annot_fn(v):
                continue
            r, g, b, _ = cmap_obj(norm(v))
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            txt = f"{v:.2f}".lstrip("0") if v < 1 else f"{v:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color="white" if lum < 0.5 else INK)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=_paths.res("memorization.csv"))
    ap.add_argument("--f", type=int, default=8, help="compression factor")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dpi", type=int, default=200)
    a, _ = ap.parse_known_args()
    out = a.out or _paths.out(f"figS_memorization_heatmap_f{a.f}.png")

    df = pd.read_csv(a.csv)
    df["fam"] = df["model"].map(fam_of)
    df = df[df["model"].map(comp_of) == a.f].copy()

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": "white",
                         "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 8.2))
    fig.suptitle("Memorization by tokenizer family × dataset",
                 fontsize=16, fontweight="bold", color=INK, y=0.98)

    specs = [
        (0, "copy_rate", COPY_CMAP, COPY_VMIN, COPY_VMAX,
         lambda v: v >= COPY_ANNOT_MIN, "Copy-rate", "copy-rate"),
        (1, "mem_ratio", RATIO_CMAP, RATIO_VMIN, RATIO_VMAX,
         lambda v: v <= RATIO_ANNOT_MAX, "NN-distance ratio", "NN-ratio (darker = tighter)"),
    ]
    for row, col, cmap, vmin, vmax, annot, rlab, cbar_lab in specs:
        ims = []
        for c, (ext, ext_title) in enumerate(EXTRACTORS):
            ax = axes[row, c]
            im = draw_heat(ax, matrix(df, ext, col), cmap, vmin, vmax, annot)
            ims.append(im)
            if row == 0:
                ax.set_title(ext_title, fontsize=13, color=INK, pad=8)
            if row == 1:
                ax.set_xticklabels([DS_LABEL[d] for d in DS_KEYS], rotation=90,
                                   fontsize=9, color=INK)
            else:
                ax.set_xticklabels([])
            if c == 1:
                ax.tick_params(labelleft=False)
        # shared colorbar for the row
        cb = fig.colorbar(ims[-1], ax=list(axes[row]), fraction=0.018, pad=0.015)
        cb.set_label(cbar_lab, fontsize=10, color=INK)
        cb.ax.tick_params(labelsize=8)
        axes[row, 0].set_ylabel(rlab, fontsize=13, fontweight="bold", color=INK,
                                labelpad=10)

    fig.subplots_adjust(left=0.10, right=0.93, top=0.90, bottom=0.14,
                        hspace=0.12, wspace=0.05)
    fig.savefig(out, dpi=a.dpi, bbox_inches="tight")
    print(f"wrote {out}  (f={a.f})")


if __name__ == "__main__":
    main()
