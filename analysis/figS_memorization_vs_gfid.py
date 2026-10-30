#!/usr/bin/env python3
"""
mem_vs_gfid.py — memorization vs generation quality (gFID), laid out like the
main memorization figure: 2x2 grid (copy-rate / NN-distance ratio) x
(SSCD / DINOv2), colored by tokenizer family. Shows that neither copy-rate nor
NN-distance ratio tracks gFID, so the memorization signal is not a
generator-underfitting artifact (rebuttal, 5CCj).
"""
import os, re
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import _paths

INK = "#1a1714"

FAM = {  # dotted key -> display family
    "aekl": "AEKL", "soft_vq": "SoftVQ", "vq": "VQGAN", "simvq": "SimVQ",
    "simple_qinco": "SimQINCo", "rqvae": "RQVAE", "diveq": "DiVeQ",
    "sf_diveq": "SF-DiVeQ", "lfq": "LFQ", "bsq": "BSQ",
}
FAM_COLOR = {  # official palette used across all figures
    "AEKL": "#5a6f8c", "SoftVQ": "#c89a3d", "VQGAN": "#b86a3c", "BSQ": "#9c5226",
    "LFQ": "#6d3a1c", "DiVeQ": "#8a6db0", "SF-DiVeQ": "#5d4585",
    "SimQINCo": "#5b8a5e", "SimVQ": "#86a86a", "RQVAE": "#1a4a5c",
}
ORDER = ["AEKL", "SoftVQ", "VQGAN", "BSQ", "LFQ", "DiVeQ", "SF-DiVeQ",
         "SimQINCo", "SimVQ", "RQVAE"]


def family_of(model):
    parts = model.split(".")
    return FAM.get(parts[1], parts[1]) if len(parts) >= 2 else model


mem = pd.read_csv(_paths.res("memorization.csv"))
tok = pd.read_csv(_paths.res("tokenizer_results.csv"))
g = tok[["model", "dataset", "gfid"]]
g = g[g["gfid"].notna() & np.isfinite(g["gfid"])]

rows = []
for ext in ("sscd", "dinov2"):
    sub = mem[mem["extractor"] == ext][["model", "dataset", "copy_rate", "mem_ratio"]]
    sub = sub.rename(columns={"copy_rate": f"copy_{ext}", "mem_ratio": f"ratio_{ext}"})
    rows.append(sub.set_index(["model", "dataset"]))
df = pd.concat(rows, axis=1).reset_index().merge(g, on=["model", "dataset"], how="inner")
df["family"] = df["model"].map(family_of)
df = df.dropna(subset=["gfid"])

plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2), sharex=True, sharey="row")

PANELS = [
    (0, 0, "copy_sscd", "log", "SSCD (copy detection)", "Copy-rate"),
    (0, 1, "copy_dinov2", "log", "DINOv2 (semantic)", None),
    (1, 0, "ratio_sscd", "linear", None, "NN-distance ratio"),
    (1, 1, "ratio_dinov2", "linear", None, None),
]

for r, c, col, scale, title, ylab in PANELS:
    ax = axes[r, c]
    d = df.dropna(subset=[col])
    for fam in ORDER:
        s = d[d["family"] == fam]
        if s.empty:
            continue
        y = s[col].clip(lower=3e-4) if scale == "log" else s[col]
        ax.scatter(s["gfid"], y, s=15, alpha=0.7, color=FAM_COLOR[fam],
                   edgecolors="none")
    rp = stats.pearsonr(d["gfid"], d[col])[0]
    ax.text(0.03, 0.05, f"$r={rp:+.2f}$", transform=ax.transAxes,
            fontsize=8.5, color=INK, va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
    if scale == "log":
        ax.set_yscale("log")
        ax.axhline(0.05, ls="--", lw=1.1, color="#c0392b")
        ax.text(0.03, 0.93, "Real-data baseline", transform=ax.transAxes,
                ha="left", va="top", fontsize=7.5, color="#c0392b")
    else:
        ax.axhline(1.0, ls="--", lw=1.1, color="#c0392b")
        ax.text(0.03, 0.93, "No single match", transform=ax.transAxes,
                ha="left", va="top", fontsize=7.5, color="#c0392b")
        ax.set_ylim(0.6, 1.03)
    if title:
        ax.set_title(title, fontsize=10, color=INK)
    if ylab:
        ax.set_ylabel(ylab, fontsize=10, color=INK)
    if r == 1:
        ax.set_xlabel("gFID $\\downarrow$", fontsize=10, color=INK)
    for sp in ax.spines.values():          # closed rectangle (all four spines)
        sp.set_color(INK); sp.set_linewidth(0.8)
    ax.grid(True, which="both", axis="y", ls=":", lw=0.5, color="#eee")

handles = [plt.Line2D([0], [0], marker="o", ls="", ms=6, color=FAM_COLOR[f],
                      label=f) for f in ORDER]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5,
           frameon=False, columnspacing=1.2, handletextpad=0.3,
           bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.subplots_adjust(wspace=0.06)          # shared y per row -> pull columns close
out = _paths.out("figS_memorization_vs_gfid.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print("Saved", out, "| n =", len(df))
for col in ["copy_sscd", "copy_dinov2", "ratio_sscd", "ratio_dinov2"]:
    d = df.dropna(subset=[col])
    print(f"  r({col}, gFID) = {stats.pearsonr(d['gfid'], d[col])[0]:+.3f}")
