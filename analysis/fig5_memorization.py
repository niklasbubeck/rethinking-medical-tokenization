# Standalone constants pulled from cell 2 (correlations) so the memorization
# cell runs on its own.
PAPER  = "#ffffff"
INK    = "#1a1714"
INK2   = "#4a443c"
INK3   = "#8a8378"
GRID   = "#d6cfbe"
ACCENT = "#a64b1f"
TEAL   = "#2d6e7e"
FAMILY_ORDER = [
    "AEKL", "MedVAE", "VAVAE", "SoftVQ", "VQGAN", "BSQ", "LFQ",
    "DiVeQ", "SF-DiVeQ", "SimQinco", "SimVQ", "RSimQinco", "RQVAE", "TiTok",
]
DOTTED_FAM_MAP = {
    "aekl": "AEKL", "medvae": "MedVAE", "vavae": "VAVAE", "soft_vq": "SoftVQ",
    "vq": "VQGAN", "bsq": "BSQ", "lfq": "LFQ", "diveq": "DiVeQ",
    "sf_diveq": "SF-DiVeQ", "simple_qinco": "SimQinco", "simvq": "SimVQ",
    "rsimple_qinco": "RSimQinco", "rqvae": "RQVAE", "titok": "TiTok",
}
#!/usr/bin/env python3
"""Memorization vs. compression figure (copy-rate + NN-distance ratio) for the
paper (Fig. 5). Reads results/memorization.csv.

  python analysis/fig5_memorization.py
"""
import re, argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import _paths

# from correlations import (
#     DOTTED_FAM_MAP, FAMILY_ORDER, FAMILY_PALETTE,
#     PAPER, INK, INK2, GRID, ACCENT,
# )

# Real held-out test data sits at copy_rate ~= 0.05 by construction (95th pct).
BASELINE = 0.05
COMP_ORDER = ["f4", "f8", "f16"]
EXTRACTORS = ["sscd", "dinov2"]
EXT_LABEL = {"sscd": "SSCD (copy detection)", "dinov2": "DINOv2 (semantic)"}

# --- paper color theme (overrides FAMILY_PALETTE from correlations.py) ---
# Keyed by the raw model token (model.split('.')[1]); re-keyed to the display
# family names via DOTTED_FAM_MAP so it matches whatever fam_of() returns.
_PAPER_THEME = {
    "aekl":         "#5a6f8c",   # blue        (continuous AE)
    "soft_vq":      "#c89a3d",   # gold
    "vq":           "#b86a3c",   # orange      (lookup-based / VQGAN)
    "bsq":          "#9c5226",   # brown       (lookup-free)
    "lfq":          "#6d3a1c",   # dark brown  (lookup-free)
    "diveq":        "#8a6db0",   # purple
    "sf_diveq":     "#5d4585",   # dark purple
    "simple_qinco": "#5b8a5e",   # green
    "simvq":        "#86a86a",   # light green
    "rqvae":        "#1a4a5c",   # dark teal
}
FAMILY_PALETTE = {DOTTED_FAM_MAP.get(k, k): v for k, v in _PAPER_THEME.items()}


def fam_of(model):
    p = model.split(".")
    return DOTTED_FAM_MAP.get(p[1], p[1]) if len(p) >= 2 else model

def comp_of(model):
    m = re.search(r"\.f(\d+)_", model)
    return f"f{m.group(1)}" if m else "?"


def load(path):
    df = pd.read_csv(path)
    df["family"] = df["model"].map(fam_of)
    df["comp"] = df["model"].map(comp_of)
    return df


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 15,
        "axes.facecolor": PAPER, "axes.linewidth": 1.6,
        "axes.edgecolor": INK, "figure.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.spines.bottom": True, "axes.spines.left": True,
    })


def _draw_swarm(ax, sub, col, baseline, show_x, fmt):
    """Within each compression group, spread the models across the x-slot; for
    each model draw its per-dataset dots (faint) plus a solid mean dot. A black
    bar marks the overall mean across all models."""
    fams = [f for f in FAMILY_ORDER if f in set(sub.family)]
    nf = len(fams)
    offs = np.linspace(-0.34, 0.34, nf) if nf > 1 else np.array([0.0])
    rng = np.random.default_rng(0)
    for xi, c in enumerate(COMP_ORDER):
        cc = sub[sub.comp == c]
        for fo, fam in zip(offs, fams):
            v = cc[cc.family == fam][col].dropna().to_numpy()
            if v.size == 0:
                continue
            x = xi + fo
            jit = rng.uniform(-0.018, 0.018, v.size)
            color = FAMILY_PALETTE.get(fam, "#888")
            ax.scatter(x + jit, v, s=32, color=color, alpha=0.30,
                       linewidths=0, zorder=2)                       # datasets
            ax.scatter([x], [np.mean(v)], s=88, color=color,
                       edgecolor="black", linewidth=0.5, zorder=4)   # model mean
        m = cc[col].mean()
        ax.plot([xi - 0.40, xi + 0.40], [m, m], color=INK, lw=2.0, zorder=5)
        ax.annotate(fmt.format(m), xy=(xi + 0.42, m), xytext=(2, 0),
                    textcoords="offset points", rotation=90, fontsize=13,
                    color=INK, va="center", ha="left", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.1", fc="white",
                              ec="none", alpha=0.7))
    ax.axhline(baseline, color=ACCENT, lw=1.4, ls=(0, (5, 3)), zorder=1)
    ax.set_xticks(range(len(COMP_ORDER)))
    if show_x:
        ax.set_xticklabels(COMP_ORDER, fontsize=16.5, color=INK)
        ax.set_xlabel("Compression factor", fontsize=16.5, color=INK)
    else:
        ax.set_xticklabels([])
    ax.grid(True, axis="y", color=GRID, lw=0.6, alpha=0.8)
    ax.tick_params(colors=INK2, labelsize=15)
    ax.set_xlim(-0.55, len(COMP_ORDER) - 1 + 0.72)


def _draw_compression_ax(ax, sub, rng, show_x, s=22, big=False):
    _draw_swarm(ax, sub, "copy_rate", BASELINE, show_x, "{:.3f}")


def _draw_ratio_ax(ax, sub, rng, show_x, s=22, big=False):
    """Somepalli NN-distance ratio d1/mean(d2..d_{k+1}); ratio 1 means no single
    memorised match, lower means a tighter copy of one specific image."""
    _draw_swarm(ax, sub, "mem_ratio", 1.0, show_x, "{:.2f}")


def fig_compression_1col(df, out, dpi):
    """Copy-rate (top) and NN-distance ratio (bottom) vs compression, one column
    per extractor. Wide enough that every point label stays readable."""
    _style()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 7.3), sharex=True, sharey="row")
    for col, ext in enumerate(EXTRACTORS):
        sub = df[df.extractor == ext]
        _draw_compression_ax(axes[0, col], sub, rng, show_x=False, s=26, big=True)
        _draw_ratio_ax(axes[1, col], sub, rng, show_x=True, s=26, big=True)
        axes[0, col].set_title(EXT_LABEL[ext], fontsize=17, color=INK, pad=6)
    axes[0, 0].set_ylabel("Copy-rate", fontsize=17, color=INK)
    axes[1, 0].set_ylabel("NN-distance ratio", fontsize=17, color=INK)
    # copy-rate spans many exact zeros + a long tail to ~0.21, so use symlog:
    # a thin linear band holds the zeros, log above spreads the small values.
    for ax in axes[0]:
        ax.set_yscale("symlog", linthresh=1e-3, linscale=0.5)
        ax.set_ylim(0, 0.30)
    axes[0, 0].text(-0.45, BASELINE, " Real-data baseline", color=ACCENT,
                    fontsize=17, va="bottom", ha="left")
    axes[1, 0].text(-0.45, 1.0, " No single match", color=ACCENT,
                    fontsize=17, va="top", ha="left")
    fams = [f for f in FAMILY_ORDER if f in set(df.family)]
    handles = [Line2D([0],[0], marker="o", ls="", markerfacecolor=FAMILY_PALETTE.get(f,"#888"),
                      markeredgecolor="black", markeredgewidth=0.5, markersize=12,
                      label=f) for f in fams]
    ncol = (len(fams) + 1) // 2          # two balanced rows
    fig.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.23,
                        hspace=0.11, wspace=0.03)
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=15,
               frameon=False, bbox_to_anchor=(0.5, 0.02),
               columnspacing=1.6, handletextpad=0.4)
    fig.savefig(out, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=_paths.res("memorization.csv"))
    ap.add_argument("--dpi", type=int, default=200)
    a, _ = ap.parse_known_args()   # ignore args Jupyter injects (e.g. --f=...kernel.json)
    df = load(a.csv)
    fig_compression_1col(df, _paths.out("fig5_memorization.png"), a.dpi)


if __name__ == "__main__":
    main()
# %%
