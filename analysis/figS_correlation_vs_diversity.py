#!/usr/bin/env python3
"""
Supplementary figure: put the per-dataset reconstruction->generation coupling in
perspective against dataset variance/diversity. For each dataset we compute the
Pearson r of rFID vs gFID (n=30 (tokenizer,compression) configs) and scatter it
against the four diversity measures of Tab. (dataset-variance). Each panel adds an
OLS fit and a second-order (r, rho) box over the 12 datasets.

  python analysis/figS_correlation_vs_diversity.py                 # rFID
"""
from __future__ import annotations
import argparse, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import _paths

TOK = _paths.res("tokenizer_results.csv")
PAPER, INK, INK2, GRID = "#ffffff", "#1a1714", "#4a443c", "#d6cfbe"
ACCENT, TEAL = "#a64b1f", "#2d6e7e"
MINUS, RHO = "−", "ρ"

DOTTED_FAM_MAP = {"aekl": "AEKL", "soft_vq": "SoftVQ", "vq": "VQGAN", "bsq": "BSQ",
                  "lfq": "LFQ", "diveq": "DiVeQ", "sf_diveq": "SF-DiVeQ",
                  "simple_qinco": "SimQinco", "simvq": "SimVQ", "rqvae": "RQVAE"}
FAMILY_ORDER = list(DOTTED_FAM_MAP.values())

DS = [("pathmnist", "Path"), ("chestmnist", "Chest"), ("dermamnist", "Derma"),
      ("octmnist", "OCT"), ("pneumoniamnist", "Pneu"), ("retinamnist", "Reti"),
      ("breastmnist", "Breast"), ("bloodmnist", "Blood"), ("tissuemnist", "Tiss"),
      ("organamnist", "OrgA"), ("organcmnist", "OrgC"), ("organsmnist", "OrgS")]

def load_diversity(csv_path):
    """label -> (pixel std, eff. rank, NN sim, pairwise dist), at table precision.

    results/dataset_variance.csv is written by table_dataset_variance.py; values
    are rounded as printed in the paper's dataset-variance table.
    """
    d = pd.read_csv(csv_path)
    return {r.label: (round(r.pixel_std, 3), round(r.eff_rank, 1), round(r.nn_sim, 3),
                      round(r.pairwise_dist, 3)) for r in d.itertuples()}


DIV = load_diversity(_paths.res("dataset_variance.csv"))
# panel spec: (col index in DIV tuple, axis label, higher-is-more-diverse arrow)
DIVCOLS = [(1, "Effective rank", "→ more diverse"),
           (0, "Pixel std", "→ more diverse"),
           (2, "NN cosine sim.", "→ more duplicates"),
           (3, "Pairwise cos. dist.", "→ more diverse")]

DS_PALETTE = plt.get_cmap("tab20")(np.linspace(0, 1, len(DS)))


def fam_of(m):
    p = str(m).split("."); return DOTTED_FAM_MAP.get(p[1], p[1]) if len(p) >= 2 else str(m)


def load():
    df = pd.read_csv(TOK)
    df = df[df.dataset != "_all_"]
    df = df[df["gfid"].notna()]
    df["family"] = df["model"].map(fam_of)
    df = df[df["family"].isin(FAMILY_ORDER)]
    return df.rename(columns={"gfid": "gFID"})


def per_dataset_r(df, key="rfid"):
    """Pearson r of `key` vs gFID within each dataset (n configs)."""
    out = {}
    for dk, dl in DS:
        s = df[df.dataset == dk]
        s = s[s[key].notna() & np.isfinite(s[key]) & s["gFID"].notna()]
        if len(s) >= 3:
            out[dl] = stats.pearsonr(s[key].to_numpy(float), s["gFID"].to_numpy(float))[0]
    return out


def _s(v):
    return ("+" if v >= 0 else MINUS) + f"{abs(v):.2f}"


def panel(ax, xvals, yvals, labels, xlabel, arrow):
    ax.set_facecolor(PAPER)
    x, y = np.asarray(xvals, float), np.asarray(yvals, float)
    for xi, yi, lb, col in zip(x, y, labels, DS_PALETTE):
        ax.scatter(xi, yi, s=70, color=col, edgecolors=INK, linewidths=0.6, zorder=3)
        ax.annotate(lb, (xi, yi), textcoords="offset points", xytext=(4.5, 3.5),
                    fontsize=7.2, color=INK2, zorder=4)
    # OLS fit + second-order stats over the 12 datasets
    xg = np.linspace(x.min(), x.max(), 50)
    b, a = np.polyfit(x, y, 1)
    ax.plot(xg, a + b * xg, color=ACCENT, lw=1.7, zorder=2)
    rp = stats.pearsonr(x, y)[0]; rs = stats.spearmanr(x, y)[0]
    ax.text(0.04, 0.05, f"$r{{=}}{_s(rp)}$\n${RHO}{{=}}{_s(rs)}$",
            transform=ax.transAxes, va="bottom", ha="left", fontsize=8,
            family="monospace", color=INK, zorder=6,
            bbox=dict(boxstyle="round,pad=0.28", fc=PAPER, ec=ACCENT, lw=0.8, alpha=0.95))
    ax.set_xlabel(f"{xlabel}  ({arrow})", fontsize=9.5, color=INK)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
    for sp in ax.spines.values():
        sp.set_edgecolor(INK); sp.set_linewidth(0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="rfid", choices=["rfid", "psnr"])
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    df = load()
    rmap = per_dataset_r(df, a.metric)
    labels = [dl for _, dl in DS if dl in rmap and dl in DIV]
    yvals = [rmap[dl] for dl in labels]
    print("per-dataset r (%s vs gFID):" % a.metric)
    for dl in labels:
        print(f"  {dl:>7s}  r={rmap[dl]:+.3f}  effrank={DIV[dl][1]}")

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": PAPER,
                         "savefig.facecolor": PAPER})
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 8.2))
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.09,
                        hspace=0.28, wspace=0.22)
    mlabel = "rFID" if a.metric == "rfid" else "PSNR"
    for ax, (ci, xlabel, arrow) in zip(axes.flat, DIVCOLS):
        xvals = [DIV[dl][ci] for dl in labels]
        panel(ax, xvals, yvals, labels, xlabel, arrow)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"per-dataset $r$ ({mlabel}$\\leftrightarrow$gFID)",
                      fontsize=9.5, color=INK)
    fig.suptitle(f"Does dataset diversity modulate the {mlabel}$\\rightarrow$gFID coupling?",
                 fontsize=13, color=INK, y=0.965)
    out = _paths.out(f"figS_correlation_vs_diversity_{a.metric}.png")
    fig.savefig(out, dpi=a.dpi, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
