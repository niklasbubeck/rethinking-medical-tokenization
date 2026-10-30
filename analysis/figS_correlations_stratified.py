#!/usr/bin/env python3
"""
Supplementary figures: the medical half of Figure 2 (rFID and PSNR vs. gFID),
faceted (a) one panel per dataset and (b) one panel per compression factor.
Each panel: scatter colored by family / marker by compression, Pearson-OLS fit
(+95% band), Spearman monotone fit, and a stat box (r, rho with bootstrap CIs).

  python analysis/figS_correlations_stratified.py   # both figures, rFID and PSNR
"""
from __future__ import annotations
import argparse, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

import _paths

TOK = _paths.res("tokenizer_results.csv")
PAPER, INK, INK2, GRID = "#ffffff", "#1a1714", "#4a443c", "#d6cfbe"
ACCENT, TEAL = "#a64b1f", "#2d6e7e"
MINUS, RHO = "−", "ρ"

DOTTED_FAM_MAP = {"aekl": "AEKL", "soft_vq": "SoftVQ", "vq": "VQGAN", "bsq": "BSQ",
                  "lfq": "LFQ", "diveq": "DiVeQ", "sf_diveq": "SF-DiVeQ",
                  "simple_qinco": "SimQinco", "simvq": "SimVQ", "rqvae": "RQVAE"}
FAMILY_ORDER = ["AEKL", "SoftVQ", "VQGAN", "BSQ", "LFQ", "DiVeQ",
                "SF-DiVeQ", "SimQinco", "SimVQ", "RQVAE"]
FAMILY_PALETTE = {"AEKL": "#5a6f8c", "SoftVQ": "#c89a3d", "VQGAN": "#b86a3c",
                  "BSQ": "#9c5226", "LFQ": "#6d3a1c", "DiVeQ": "#8a6db0",
                  "SF-DiVeQ": "#5d4585", "SimQinco": "#5b8a5e", "SimVQ": "#86a86a",
                  "RQVAE": "#1a4a5c"}
MED_MARKER = {"f4": "o", "f8": "D", "f16": "^"}
# metric column -> (df key, axis label, unit, higher_is_better)
METRIC = {"rFID": ("rfid", "rFID", "", False), "PSNR": ("psnr", "PSNR", " (dB)", True)}

DS = [("pathmnist", "Path"), ("chestmnist", "Chest"), ("dermamnist", "Derma"),
      ("octmnist", "OCT"), ("pneumoniamnist", "Pneu"), ("retinamnist", "Reti"),
      ("breastmnist", "Breast"), ("bloodmnist", "Blood"), ("tissuemnist", "Tiss"),
      ("organamnist", "OrgA"), ("organcmnist", "OrgC"), ("organsmnist", "OrgS")]
COMPS = ["f4", "f8", "f16"]


def fam_of(m):
    p = str(m).split("."); return DOTTED_FAM_MAP.get(p[1], p[1]) if len(p) >= 2 else str(m)


def load():
    df = pd.read_csv(TOK)
    df = df[df.dataset != "_all_"]
    df = df[df["gfid"].notna()]
    df["family"] = df["model"].map(fam_of)
    df = df[df["family"].isin(FAMILY_ORDER)]
    df["compression"] = df["model"].map(lambda m: "f" + re.search(r"\.f(\d+)_", str(m)).group(1))
    return df.rename(columns={"gfid": "gFID"})


def _spear_curve(x, y, n=64):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    b, a = np.polyfit(rx, ry, 1)
    xs, ys, ranks = np.sort(x), np.sort(y), np.arange(1, len(x) + 1)
    xv = np.linspace(x.min(), x.max(), n)
    return xv, np.interp(np.clip(a + b * np.interp(xv, xs, ranks), 1, len(x)), ranks, ys)


def _ols_band(x, y, xg):
    x, y, n = np.asarray(x, float), np.asarray(y, float), len(x)
    b, a = np.polyfit(x, y, 1)
    s2 = np.sum((y - (a + b * x)) ** 2) / (n - 2)
    xb, sxx = x.mean(), np.sum((x - x.mean()) ** 2)
    se = np.sqrt(s2 * (1 / n + (xg - xb) ** 2 / sxx))
    return a + b * xg, stats.t.ppf(0.975, n - 2) * se


def _ci(x, y, fn, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(x); out = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n); out[i] = fn(x[idx], y[idx])
    return np.nanpercentile(out, [2.5, 97.5])


def _s(v):
    return ("+" if v >= 0 else MINUS) + f"{abs(v):.2f}"


def facet(ax, sub, key, hb, title, small=True):
    sub = sub[sub[key].notna() & np.isfinite(sub[key]) & sub["gFID"].notna()]
    x, y = sub[key].to_numpy(float), sub["gFID"].to_numpy(float)
    ax.set_facecolor(PAPER)
    for fam in FAMILY_ORDER:
        s = sub[sub.family == fam]
        for comp, cd in s.groupby("compression"):
            ax.scatter(cd[key], cd["gFID"], marker=MED_MARKER.get(comp, "o"),
                       s=16 if small else 34, c=FAMILY_PALETTE[fam],
                       edgecolors=PAPER, linewidths=0.3, alpha=0.85, zorder=3)
    if len(x) >= 3:
        xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
        xpad, ypad = (xmax - xmin) * 0.06 or 1, (ymax - ymin) * 0.06 or 1
        xlo, xhi = (xmin - xpad, xmax + xpad) if hb else (max(0, xmin - xpad), xmax + xpad)
        ylo, yhi = max(0, ymin - ypad), ymax + ypad
        xg = np.linspace(xlo, xhi, 100); fit, half = _ols_band(x, y, xg)
        ax.fill_between(xg, fit - half, fit + half, color=ACCENT, alpha=0.13, lw=0, zorder=2)
        b, a = np.polyfit(x, y, 1); yl = a + b * xg; m = (yl >= ylo) & (yl <= yhi)
        ax.plot(xg[m], yl[m], color=ACCENT, lw=1.6, zorder=4)
        xv, yv = _spear_curve(x, y); m2 = (yv >= ylo) & (yv <= yhi)
        ax.plot(xv[m2], yv[m2], color=TEAL, lw=1.4, ls=(0, (4, 3)), zorder=4)
        rp = stats.pearsonr(x, y)[0]; lp, hp = _ci(x, y, lambda a, b: stats.pearsonr(a, b)[0])
        rs = stats.spearmanr(x, y)[0]
        ax.set_ylim(yhi, ylo); ax.set_xlim((xlo, xhi) if hb else (xhi, xlo))
        ax.text(0.04, 0.05, f"$r{{=}}{_s(rp)}$ [{_s(lp)},{_s(hp)}]\n${RHO}{{=}}{_s(rs)}$",
                transform=ax.transAxes, va="bottom", ha="left", fontsize=6.6,
                family="monospace", color=INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", fc=PAPER, ec=ACCENT, lw=0.7, alpha=0.95))
    ax.set_title(f"{title}  ($n{{=}}{len(sub)}$)", fontsize=10, color=INK, pad=3)
    ax.tick_params(colors=INK2, labelsize=7.5)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.7)
    for sp in ax.spines.values():
        sp.set_edgecolor(INK); sp.set_linewidth(0.9)


def legend(fig, y=0.02, ncol=8, fontsize=8.5):
    h = [Line2D([0], [0], marker="o", ls="", ms=6, mfc=FAMILY_PALETTE[f], mec=PAPER, label=f)
         for f in FAMILY_ORDER]
    h += [Line2D([0], [0], marker=MED_MARKER[c], ls="", ms=6, mfc=INK, mec=PAPER, label=c) for c in COMPS]
    h += [Line2D([0], [0], color=ACCENT, lw=1.8, label="Pearson (OLS)"),
          Line2D([0], [0], color=TEAL, lw=1.6, ls=(0, (4, 3)), label=f"Spearman {RHO}")]
    lg = fig.legend(handles=h, loc="lower center", ncol=ncol, frameon=False, fontsize=fontsize,
                    bbox_to_anchor=(0.5, y), columnspacing=1.1, handletextpad=0.4)
    for t in lg.get_texts():
        t.set_color(INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()
    df = load()
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": PAPER,
                         "savefig.facecolor": PAPER})
    metrics = [(m, *METRIC[m]) for m in ("rFID", "PSNR")]   # (name, key, label, unit, hb)

    # ---- (a) per dataset: 3x(4|4) with a wide central gutter, both metrics ----
    fig, axes = plt.subplots(3, 9, figsize=(24.5, 9.8),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.55, 1, 1, 1, 1]})
    fig.subplots_adjust(left=0.04, right=0.996, top=0.915, bottom=0.10,
                        hspace=0.34, wspace=0.30)
    for ax in axes[:, 4]:                              # central spacer column
        ax.axis("off")
    blocks = [axes[:, 0:4], axes[:, 5:9]]
    for mi, (mname, key, label, unit, hb) in enumerate(metrics):
        arrow = "↑" if hb else "↓"
        block = blocks[mi]
        for ax, (dk, dl) in zip(block.flat, DS):
            facet(ax, df[df.dataset == dk], key, hb, dl)
        for ax in block[-1]:
            ax.set_xlabel(f"{label}{unit} {arrow}", fontsize=8.5, color=INK)
        for ax in block[:, 0]:
            ax.set_ylabel("gFID $\\downarrow$", fontsize=8.5, color=INK)
    # headers + divider centered on the actual block/gutter positions
    lx = (axes[0, 0].get_position().x0 + axes[0, 3].get_position().x1) / 2
    rx = (axes[0, 5].get_position().x0 + axes[0, 8].get_position().x1) / 2
    dx = (axes[0, 3].get_position().x1 + axes[0, 5].get_position().x0) / 2
    fig.text(lx, 0.95, "rFID vs. gFID per dataset", ha="center", fontsize=14, color=INK)
    fig.text(rx, 0.95, "PSNR vs. gFID per dataset", ha="center", fontsize=14, color=INK)
    fig.add_artist(Line2D([dx, dx], [0.06, 0.925], transform=fig.transFigure,
                          color=INK, lw=1.3))
    legend(fig, y=0.015, ncol=15, fontsize=9)
    out1 = _paths.out("figS_correlations_perdataset.png")
    fig.savefig(out1, dpi=a.dpi, bbox_inches="tight"); plt.close(fig)
    print("wrote", out1)

    # ---- (b) per compression: 2x3 (rFID row / PSNR row), both metrics combined ----
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.8))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.90, bottom=0.15,
                        hspace=0.30, wspace=0.20)
    for mi, (mname, key, label, unit, hb) in enumerate(metrics):
        arrow = "↑" if hb else "↓"
        for ax, c in zip(axes[mi], COMPS):
            facet(ax, df[df.compression == c], key, hb, f"$f{{=}}{c[1:]}$", small=False)
            ax.set_xlabel(f"{label}{unit} {arrow}", fontsize=10.5, color=INK)
        axes[mi][0].set_ylabel("gFID $\\downarrow$", fontsize=11, color=INK)
    fig.text(0.016, 0.715, "rFID", rotation=90, va="center", ha="center",
             fontsize=14, color=INK)
    fig.text(0.016, 0.30, "PSNR", rotation=90, va="center", ha="center",
             fontsize=14, color=INK)
    fig.suptitle("Reconstruction vs. gFID per compression factor",
                 fontsize=14, color=INK, y=0.965)
    legend(fig, y=0.01)
    out2 = _paths.out("figS_correlations_percompression.png")
    fig.savefig(out2, dpi=a.dpi, bbox_inches="tight"); plt.close(fig)
    print("wrote", out2)


if __name__ == "__main__":
    main()
