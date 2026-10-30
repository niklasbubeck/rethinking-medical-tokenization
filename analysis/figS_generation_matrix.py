#!/usr/bin/env python3
"""
gen_matrix.py — qualitative GENERATION comparison. Unlike recon_matrix there is
no ground truth: each cell is an unconditional DiT sample. We show n samples per
dataset for every tokenizer family at a given compression, laid out as a
(family x [dataset x n-samples]) matrix.

    # all datasets, all models, 3 samples each, f=8 (defaults)
    python gen_matrix.py

    # subset of datasets / models, 2 samples, f=16
    python gen_matrix.py --datasets octmnist pathmnist --models RQVAE LFQ BSQ \
        --n 2 --f 16 --out gen_oct_path.png
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


def sample_dir(model, ds):
    return os.path.join(BASE, f"dit_{model}", "generation", ds, "sample")

def list_samples(model, ds, n):
    """First n generated PNGs (sorted) for a (model, dataset), as full paths."""
    d = sample_dir(model, ds)
    if not os.path.isdir(d):
        return []
    fs = sorted(f for f in os.listdir(d) if f.endswith(".png"))
    return [os.path.join(d, f) for f in fs[:n]]


def load_img(p):
    im = imread(p)
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[..., :3]
    return im

def make(out_path, datasets=None, model_names=None, f=8, n=3, dpi=200):
    datasets = datasets or ALL_DS
    model_names = model_names or ALL_FAMS
    fam2model = dict(models_for(f))
    unknown = [m for m in model_names if m not in fam2model]
    if unknown:
        raise SystemExit(f"unknown model(s): {unknown}\nchoose from: {ALL_FAMS}")
    # rows = families; cols = datasets x n-samples
    rows = [(m, fam2model[m]) for m in model_names]
    cols = [(dk, DS_LABEL.get(dk, dk), k) for dk in datasets for k in range(n)]
    HEADER_FS, ROW_FS = 13, 12
    nr, nc = len(rows), len(cols)

    # cache the per-(model, dataset) sample list so we glob each dir once
    samp = {(model, dk): list_samples(model, dk, n)
            for (_, model) in rows for dk in datasets}

    plt.rcParams.update({"font.family":"DejaVu Sans","figure.facecolor":"white",
                         "savefig.facecolor":"white"})
    fig, axes = plt.subplots(nr, nc, figsize=(nc*1.05, nr*1.05), squeeze=False)
    fig.subplots_adjust(left=0.04, right=0.998, top=0.955, bottom=0.003,
                        wspace=0.0, hspace=0.0)

    for i, (rname, model) in enumerate(rows):
        for j, (dk, dlabel, k) in enumerate(cols):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#d6cfbe"); s.set_linewidth(0.5)
            files = samp[(model, dk)]
            p = files[k] if k < len(files) else None
            if p and os.path.isfile(p):
                ax.imshow(load_img(p))
            else:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center",
                        va="center", fontsize=6, color="#bbb")
            if j == 0:
                ax.set_ylabel(rname, fontsize=ROW_FS, color=INK, fontweight="bold",
                              rotation=90, ha="center", va="center", labelpad=4)

    # centred dataset header spanning each sample-column group
    for di, dk in enumerate(datasets):
        a0 = axes[0, di * n].get_position()
        a1 = axes[0, di * n + n - 1].get_position()
        fig.text(0.5 * (a0.x0 + a1.x1), a0.y1 + 0.006, DS_LABEL.get(dk, dk),
                 ha="center", va="bottom", fontsize=HEADER_FS,
                 fontweight="bold", color=INK)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out_path}  (f={f}, n={n})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=ALL_DS,
                    help=f"subset/order of datasets (default all). Choices: {ALL_DS}")
    ap.add_argument("--models", nargs="+", default=ALL_FAMS,
                    help=f"subset/order of families (default all). Choices: {ALL_FAMS}")
    ap.add_argument("--n", type=int, default=3, help="number of samples per dataset")
    ap.add_argument("--f", type=int, default=8, choices=[4,8,16], help="compression factor")
    ap.add_argument("--out", default=None, help="default: analysis_out/figS_generation_matrix_f<f>.png")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    a.out = a.out or _paths.out(f"figS_generation_matrix_f{a.f}.png")
    make(a.out, a.datasets, a.models, a.f, a.n, a.dpi)
