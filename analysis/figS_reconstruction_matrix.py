#!/usr/bin/env python3
"""
recon_matrix.py — qualitative reconstruction comparison. One fixed test image
("subject") per dataset, reconstructed by the chosen tokenizer families at a
given compression, laid out as a (dataset x [Original | models]) matrix.

    # all datasets, all models, subject #7, f=8 (defaults)
    python recon_matrix.py

    # subset of datasets / models, a specific subject, f=16
    python recon_matrix.py --datasets octmnist pathmnist --models RQVAE LFQ BSQ \
        --idx 42 --f 16 --out recon_oct_path.png
"""
import os, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

import _paths

BASE = _paths.RUNS   # pipeline outputs ($RMT_BASE_DIR/outputs/$RMT_PROJECT)
INK = "#1a1714"

DS_ORDER = [("pathmnist","Path"),("chestmnist","Chest"),("dermamnist","Derma"),
            ("octmnist","OCT"),("pneumoniamnist","Pneu"),("retinamnist","Reti"),
            ("breastmnist","Breast"),("bloodmnist","Blood"),("tissuemnist","Tiss"),
            ("organamnist","OrgA"),("organcmnist","OrgC"),("organsmnist","OrgS")]
DS_LABEL = dict(DS_ORDER)
ALL_DS   = [k for k, _ in DS_ORDER]
ALL_FAMS = ["AEKL","SoftVQ","VQGAN","SimVQ","SimQINCo","RQVAE","DiVeQ","SF-DiVeQ","BSQ","LFQ"]

# family -> dotted model name, parameterised by compression f
def models_for(f):
    e = "e8192" if f == 4 else "e16384"
    b = {4: "b10", 8: "b14", 16: "b18"}[f]
    dlf = {4: "d10", 8: "d14", 16: "d18"}[f]
    dd = {4: "d3", 8: "d4", 16: "d8"}[f]
    return [
        ("AEKL",     f"continuous.aekl.f{f}_{dd}"),
        ("SoftVQ",   f"continuous.soft_vq.f{f}_{dd}_{e}"),
        ("VQGAN",    f"discrete.vq.f{f}_{dd}_{e}"),
        ("SimVQ",    f"discrete.simvq.f{f}_{dd}_{e}"),
        ("SimQINCo", f"discrete.simple_qinco.f{f}_{dd}_{e}"),
        ("RQVAE",    f"discrete.rqvae.f{f}_{dd}_{e}"),
        ("DiVeQ",    f"discrete.diveq.f{f}_{dd}_{e}"),
        ("SF-DiVeQ", f"discrete.sf_diveq.f{f}_{dd}_{e}"),
        ("BSQ",      f"discrete.bsq.f{f}_{dlf}_{b}"),
        ("LFQ",      f"discrete.lfq.f{f}_{dlf}_{b}"),
    ]

# per-dataset subject index override (default = --idx for all)
SUBJECT = {}


def cell_path(model, ds, idx):
    return os.path.join(BASE, model, "reconstruction", ds, "sample", f"{idx:08d}.png")

def ref_path(ds, idx):
    return os.path.join(BASE, "reference", ds, "reference", f"{idx:08d}.png")


def load_img(p):
    im = imread(p)                       # PNG -> float32 in [0, 1]
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[..., :3]                 # drop alpha
    return im

def psnr(ref, rec):
    """PSNR (dB) between reference and reconstruction, both float in [0, 1]."""
    if ref is None or rec is None or ref.shape != rec.shape:
        return None
    mse = float(np.mean((ref.astype(np.float64) - rec.astype(np.float64)) ** 2))
    if mse <= 0:
        return float("inf")
    return -10.0 * np.log10(mse)         # MAX = 1.0 -> 20*log10(1) = 0


def make(out_path, datasets=None, model_names=None, f=8, idx=7, dpi=200):
    datasets = datasets or ALL_DS
    model_names = model_names or ALL_FAMS
    subjects = idx if isinstance(idx, (list, tuple)) else [idx]
    fam2model = dict(models_for(f))
    unknown = [m for m in model_names if m not in fam2model]
    if unknown:
        raise SystemExit(f"unknown model(s): {unknown}\nchoose from: {ALL_FAMS}")
    # Inverted layout: rows = [Original | models], cols = datasets x subjects
    # (each subject adds another column per dataset, extending sideways)
    rows = [("Original", None)] + [(m, fam2model[m]) for m in model_names]
    cols = [(dk, DS_LABEL.get(dk, dk), s) for dk in datasets for s in subjects]
    nsub = len(subjects)
    HEADER_FS, ROW_FS = 13, 12          # column (dataset) and row (family) label sizes
    nr, nc = len(rows), len(cols)
    plt.rcParams.update({"font.family":"DejaVu Sans","figure.facecolor":"white",
                         "savefig.facecolor":"white"})
    fig, axes = plt.subplots(nr, nc, figsize=(nc*1.05, nr*1.05), squeeze=False)
    fig.subplots_adjust(left=0.04, right=0.998, top=0.955, bottom=0.003,
                        wspace=0.0, hspace=0.0)

    ref_cache = {}
    for i, (rname, model) in enumerate(rows):
        for j, (dk, dlabel, si) in enumerate(cols):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#d6cfbe"); s.set_linewidth(0.5)
            p = ref_path(dk, si) if model is None else cell_path(model, dk, si)
            if os.path.isfile(p):
                img = load_img(p)
                ax.imshow(img)
                if model is not None:    # annotate PSNR vs the reference subject
                    rp = ref_path(dk, si)
                    if (dk, si) not in ref_cache:
                        ref_cache[(dk, si)] = load_img(rp) if os.path.isfile(rp) else None
                    val = psnr(ref_cache[(dk, si)], img)
                    if val is not None and np.isfinite(val):
                        ax.text(0.96, 0.94, f"{val:.1f}", transform=ax.transAxes,
                                ha="right", va="top", fontsize=5.0, color="white",
                                fontweight="bold", zorder=5,
                                bbox=dict(boxstyle="round,pad=0.12",
                                          facecolor="black", alpha=0.45,
                                          edgecolor="none"))
            else:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center",
                        va="center", fontsize=6, color="#bbb")
            if model is None:   # outline the original row
                for s in ax.spines.values():
                    s.set_color(INK); s.set_linewidth(1.3)
            if j == 0:
                ax.set_ylabel(rname, fontsize=ROW_FS, color=INK, fontweight="bold",
                              rotation=90, ha="center", va="center", labelpad=4)

    # centred dataset header spanning each subject-column group
    for di, dk in enumerate(datasets):
        a0 = axes[0, di * nsub].get_position()
        a1 = axes[0, di * nsub + nsub - 1].get_position()
        fig.text(0.5 * (a0.x0 + a1.x1), a0.y1 + 0.006, DS_LABEL.get(dk, dk),
                 ha="center", va="bottom", fontsize=HEADER_FS,
                 fontweight="bold", color=INK)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out_path}  (f={f}, subjects={subjects})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ALL_DS,
                    help=f"subset/order of datasets (default all). Choices: {ALL_DS}")
    ap.add_argument("--models", nargs="+", default=ALL_FAMS,
                    help=f"subset/order of families (default all). Choices: {ALL_FAMS}")
    ap.add_argument("--idx", type=int, nargs="+", default=[7],
                    help="one or more test-image indices (subjects); each adds a "
                         "column per dataset")
    ap.add_argument("--f", type=int, default=8, choices=[4,8,16], help="compression factor")
    ap.add_argument("--out", default=None, help="default: analysis_out/figS_reconstruction_matrix_f<f>.png")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    a.out = a.out or _paths.out(f"figS_reconstruction_matrix_f{a.f}.png")
    make(a.out, a.datasets, a.models, a.f, a.idx, a.dpi)
