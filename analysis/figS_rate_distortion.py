#!/usr/bin/env python3
"""
rate_distortion.py — rate-distortion view of the tokenizers at f=8.
Plots latent quantization error D against the bits actually used H(C), so
methods are compared as operating points rather than by the single ratio
D/H(C) (which is not scale-invariant across rates). Reads the aggregated CSV
written by the misalignment recompute.
"""
import os, csv
import numpy as np
import matplotlib.pyplot as plt

import _paths

CSV = _paths.res("dperbit_entropy_f8.csv")
INK = "#1a1714"
COLOR = {
    "VQGAN": "#8c6d31", "SimVQ": "#4e79a7", "SimQINCo": "#59a14f",
    "RQVAE": "#e15759", "DiVeQ": "#76b7b2", "SF-DiVeQ": "#edc948",
    "LFQ": "#b07aa1", "BSQ": "#ff9da7",
}


def load():
    rows = []
    with open(CSV) as f:
        for r in csv.DictReader(f):
            rows.append((r["family"], float(r["B_bits"]), float(r["H_C_bits"]),
                         float(r["D"]), float(r["D_per_H_new"])))
    return rows


def main():
    rows = load()
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    plt.rcParams.update({"font.family": "DejaVu Sans"})

    # faint iso-D/H(C) guide lines (constant distortion-per-used-bit): D = c * H
    H = np.linspace(8, 60, 50)
    for c in [1e-8, 1e-6, 1e-4, 1e-2, 1e0]:
        ax.plot(H, c * H, color="#d6cfbe", lw=0.7, zorder=0)

    # per-family label offsets (points) to avoid overlap in the central cluster
    OFF = {"VQGAN": (7, -12), "SimVQ": (7, 6), "SimQINCo": (10, -2),
           "DiVeQ": (8, -2), "RQVAE": (9, 0), "SF-DiVeQ": (9, 2),
           "LFQ": (9, 2), "BSQ": (9, 2)}
    for fam, B, Hc, D, dph in rows:
        col = COLOR.get(fam, INK)
        ax.scatter(Hc, D, s=90, color=col, edgecolors=INK, linewidths=0.6, zorder=3)
        ax.annotate(fam, (Hc, D), textcoords="offset points",
                    xytext=OFF.get(fam, (7, 3)),
                    fontsize=8.5, color=INK, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xlabel("bits actually used  $H(\\mathcal{C}) = \\sum_\\ell H(\\mathcal{C}_\\ell)$",
                  fontsize=10.5, color=INK)
    ax.set_ylabel("latent quantization error  $\\mathcal{D}$", fontsize=10.5, color=INK)
    ax.set_title("Rate-distortion at $f{=}8$: distortion vs. bits used",
                 fontsize=11.5, color=INK)
    ax.text(0.98, 0.03, "diagonal guides = constant $\\mathcal{D}/H(\\mathcal{C})$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            color="#9a927f")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, which="both", axis="y", ls=":", lw=0.5, color="#eee", zorder=0)
    fig.tight_layout()
    out = _paths.out("figS_rate_distortion_f8.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("Saved", out)


if __name__ == "__main__":
    main()
