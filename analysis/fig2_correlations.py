#!/usr/bin/env python3
"""
Figure 2: reconstruction-metric vs generation-FID correlation row.

Left half  : Natural imaging  (data from natural.csv, wide spreadsheet format)
Right half : Medical imaging   (data from the combined tokenizer_results.csv)

Each panel shows:
  - scatter coloured by model/family, marker = compression factor
  - Pearson OLS fit with a 95 % confidence band
  - Spearman rank fit (monotone) with a bootstrap confidence band
  - a stat box with Pearson r / Spearman rho and bootstrap 95 % CIs
  - optional top/right marginal distributions (--marginals)

Natural-image numbers are those reported by Xu et al. (2026), "Making
Reconstruction FID Predictive of Diffusion Generation FID" (results/natural.csv).

Usage:
    python analysis/fig2_correlations.py                 # paper figure
    python analysis/fig2_correlations.py --no_marginals  # without marginals
"""
from __future__ import annotations
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats

import _paths

# --------------------------------------------------------------------------- #
# Palette / aesthetics
# --------------------------------------------------------------------------- #
PAPER = "#ffffff"
INK   = "#1a1714"
INK2  = "#4a443c"
INK3  = "#8a8378"
GRID  = "#d6cfbe"
ACCENT = "#a64b1f"   # Pearson (OLS)
TEAL   = "#2d6e7e"   # Spearman (rank)
MARG_FILL = "#dcdcdc"   # light gray, shared by stat box + marginal bars
MARG_LINE = INK2
MINUS = "\u2212"
RHO = "\u03c1"

# the two panels drawn per half: (column-key, label, unit, higher-is-better)
PANELS = [("PSNR", "PSNR", " (dB)", True), ("rFID", "rFID", "", False)]

# --------------------------------------------------------------------------- #
# Medical loader  (combined tokenizer_results.csv -> one row per tokenizer x dataset)
# --------------------------------------------------------------------------- #
DOTTED_FAM_MAP = {
    "aekl": "AEKL", "medvae": "MedVAE", "vavae": "VAVAE", "soft_vq": "SoftVQ",
    "vq": "VQGAN", "bsq": "BSQ", "lfq": "LFQ", "diveq": "DiVeQ",
    "sf_diveq": "SF-DiVeQ", "simple_qinco": "SimQinco", "simvq": "SimVQ",
    "rsimple_qinco": "RSimQinco", "rqvae": "RQVAE", "titok": "TiTok",
}
EXCLUDED_FAMILIES = {"RSimQinco", "VAVAE", "TiTok"}
FAMILY_ORDER = ["AEKL", "SoftVQ", "VQGAN", "BSQ", "LFQ", "DiVeQ",
                "SF-DiVeQ", "SimQinco", "SimVQ", "RQVAE"]
FAMILY_PALETTE = {
    "AEKL": "#5a6f8c", "SoftVQ": "#c89a3d", "VQGAN": "#b86a3c", "BSQ": "#9c5226",
    "LFQ": "#6d3a1c", "DiVeQ": "#8a6db0", "SF-DiVeQ": "#5d4585",
    "SimQinco": "#5b8a5e", "SimVQ": "#86a86a", "RQVAE": "#1a4a5c",
}
MED_MARKER = {"f4": "o", "f8": "D", "f16": "^"}


def _family_of_key(dotted: str) -> str:
    parts = str(dotted).split(".")
    key = parts[1] if len(parts) >= 2 else str(dotted)
    return DOTTED_FAM_MAP.get(key, key)


def load_medical(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rows = []
    for _, r in df.iterrows():
        gfid = r.get("gfid")
        if pd.isna(gfid):
            continue
        family = _family_of_key(str(r["model"]))
        if family in EXCLUDED_FAMILIES:
            continue
        m = re.search(r"\.f(\d+)_", str(r["model"]))
        compression = f"f{m.group(1)}" if m else "?"
        rows.append({
            "color_key": family, "compression": compression,
            "PSNR": r.get("psnr"), "LPIPS": r.get("lpips"),
            "rFID": r.get("rfid"), "gFID": float(gfid),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Natural loader  (natural.csv -> wide spreadsheet, multi-row header)
# --------------------------------------------------------------------------- #
# Column indices in natural.csv (0-based, data starts at spreadsheet row 4):
#   0 = VAE Name,  7 = -PSNR (stored negated),  8 = LPIPS,  10 = rFID,
#   26 = Compression f.  gFID has six variants; the reference figure uses
#   column 3 = "gFID SiT/XL w/o CFG" (reproduces its published correlations).
NATURAL_GFID_COL = 3   # gFID SiT/XL w/o CFG   (alt: 1 SiT/B w/o, 5 SD-UNet w/o, ...)

# One colour per model.  Edit freely to match your paper exactly.
NATURAL_PALETTE = {
    "SD-VAE": "#1f77b4", "SD3-VAE": "#7fb1d3", "FLUX-VAE": "#e8843c",
    "DC-AE": "#21618c", "SOFT-VQ": "#2ca02c", "VA-VAE": "#98c17a",
    "VA-VAE-64": "#d62728", "EQ-VAE": "#f4a0a0", "MAE-TOK": "#8e6fb0",
    "IN-VAE": "#3a9e8f", "REPAE-SDVAE": "#b58a5a", "DE-TOK": "#7b4b2a",
    "QwenImg-VAE": "#d44fa0", "RAE": "#e2b73e", "SVG": "#4d4d4d",
    "FLUX2-VAE": "#9aa0a6", "SVG T2I": "#b5c44a", "DM-VAE": "#cbb079",
    "UAE": "#3bb3c4", "VTP-S": "#9fb0e0", "VTP-L": "#3b3b8f",
    "PAE DINOv2": "#4a78c2",
}
NAT_MARKER = {"f16": "o", "f32": "D", "f45.25": "^", "f64": "s"}


def _comp_label(v) -> str:
    if pd.isna(v):
        return "?"
    v = float(v)
    return f"f{int(v)}" if v.is_integer() else f"f{v:g}"


def load_natural(csv_path: str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, header=None)
    data = raw.iloc[3:].reset_index(drop=True)   # skip empty + 2 header rows

    def num(c):
        return pd.to_numeric(data[c], errors="coerce")

    df = pd.DataFrame({
        "color_key": data[0].astype(str).str.strip(),
        "PSNR": -num(7),                       # column is stored as -PSNR
        "LPIPS": num(8),
        "rFID": num(10),
        "gFID": num(NATURAL_GFID_COL),
        "compression": pd.to_numeric(data[26], errors="coerce").map(_comp_label),
    })
    return df.dropna(subset=["PSNR", "gFID"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------- #
def spearman_monotone_curve(x, y, n_pts=64):
    """Spearman trend: OLS in rank space mapped back to data space (monotone)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = stats.rankdata(x); ry = stats.rankdata(y)
    b, a = np.polyfit(rx, ry, 1)
    xs, ys = np.sort(x), np.sort(y); n = len(x)
    ranks = np.arange(1, n + 1)
    xv = np.linspace(x.min(), x.max(), n_pts)
    rxv = np.interp(xv, xs, ranks)
    yhat = np.interp(np.clip(a + b * rxv, 1, n), ranks, ys)
    return xv, yhat


def ols_ci_band(x, y, xg, alpha=0.05):
    """95% confidence band for the OLS mean prediction."""
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    b, a = np.polyfit(x, y, 1)
    s2 = np.sum((y - (a + b * x)) ** 2) / (n - 2)
    xbar = x.mean(); sxx = np.sum((x - xbar) ** 2)
    se = np.sqrt(s2 * (1.0 / n + (xg - xbar) ** 2 / sxx))
    t = stats.t.ppf(1 - alpha / 2, n - 2)
    return a + b * xg, t * se


def spearman_boot_band(x, y, xg, B=1000, seed=0):
    """Bootstrap band for the Spearman monotone curve over grid xg."""
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    rng = np.random.default_rng(seed)
    curves = np.full((B, len(xg)), np.nan)
    for i in range(B):
        idx = rng.integers(0, n, n)
        try:
            xv, yv = spearman_monotone_curve(x[idx], y[idx])
            o = np.argsort(xv)
            curves[i] = np.interp(xg, xv[o], yv[o], left=np.nan, right=np.nan)
        except Exception:
            pass
    return (np.nanpercentile(curves, 2.5, axis=0),
            np.nanpercentile(curves, 97.5, axis=0))


def boot_corr_ci(x, y, fn, B=5000, seed=0):
    """Bootstrap percentile 95% CI for a correlation coefficient."""
    rng = np.random.default_rng(seed); n = len(x); out = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        out[i] = fn(x[idx], y[idx])
    return np.percentile(out, 2.5), np.percentile(out, 97.5)


def _s3(v):  # signed, 3 dp
    return ("+" if v >= 0 else MINUS) + f"{abs(v):.3f}"


def _s2(v):  # signed, 2 dp
    return ("+" if v >= 0 else MINUS) + f"{abs(v):.2f}"


# --------------------------------------------------------------------------- #
# Marginal strip
# --------------------------------------------------------------------------- #
def _marginal(ax_m, data, lo, hi, horizontal=False):
    data = np.asarray(data, float); data = data[np.isfinite(data)]
    bins = np.linspace(min(data.min(), lo), max(data.max(), hi), 24)
    kw = dict(density=True, color=MARG_FILL, edgecolor=PAPER, linewidth=0.4, zorder=2)
    ax_m.hist(data, bins=bins, orientation="horizontal" if horizontal else "vertical", **kw)
    try:
        kde = stats.gaussian_kde(data)
        grid = np.linspace(data.min(), data.max(), 200); d = kde(grid)
        ax_m.plot(*((d, grid) if horizontal else (grid, d)), color=MARG_LINE, lw=1.6, zorder=3)
    except Exception:
        pass
    ax_m.set_facecolor(PAPER); ax_m.grid(False)
    for s in ax_m.spines.values():
        s.set_visible(False)
    ax_m.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


# --------------------------------------------------------------------------- #
# One panel
# --------------------------------------------------------------------------- #
def draw_panel(ax, df, key, label, unit, higher_better,
               palette, marker_map, marginals, seed, n_left=False):
    sub = df[df[key].notna() & np.isfinite(df[key])
             & df["gFID"].notna() & np.isfinite(df["gFID"])]
    x = sub[key].to_numpy(float); y = sub["gFID"].to_numpy(float)
    ax.set_facecolor(PAPER)

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    xpad = (xmax - xmin) * 0.06 or 1.0
    ypad = (ymax - ymin) * 0.06 or 1.0
    x_lo, x_hi = (xmin - xpad, xmax + xpad) if higher_better else (max(0, xmin - xpad), xmax + xpad)
    y_lo, y_hi = max(0, ymin - ypad), ymax + ypad

    ax_top = ax_right = None
    if marginals:
        div = make_axes_locatable(ax)
        ax_top = div.append_axes("top", size="15%", pad=0.05, sharex=ax)
        ax_right = div.append_axes("right", size="15%", pad=0.05, sharey=ax)

    present = []
    for ck, cdf in sub.groupby("color_key"):
        color = palette.get(ck, "#888888")
        for comp, ccdf in cdf.groupby("compression"):
            ax.scatter(ccdf[key], ccdf["gFID"], marker=marker_map.get(comp, "o"),
                       s=90, c=color, edgecolors=PAPER, linewidths=0.9, alpha=0.85, zorder=3)
        if ck not in present:
            present.append(ck)

    rp = stats.pearsonr(x, y)[0]; rs = stats.spearmanr(x, y)[0]
    lp, hp = boot_corr_ci(x, y, lambda a, b: stats.pearsonr(a, b)[0], seed=seed)
    ls, hs = boot_corr_ci(x, y, lambda a, b: stats.spearmanr(a, b)[0], seed=seed + 1)

    # Pearson OLS + band
    b, a = np.polyfit(x, y, 1)
    xg = np.linspace(x_lo, x_hi, 200)
    fit, half = ols_ci_band(x, y, xg)
    ax.fill_between(xg, fit - half, fit + half, color=ACCENT, alpha=0.15, lw=0, zorder=2)
    xl = np.linspace(x_lo, x_hi, 50); yl = a + b * xl
    m = (yl >= y_lo) & (yl <= y_hi)
    ax.plot(xl[m], yl[m], color=ACCENT, lw=2.6, alpha=0.95, zorder=4)

    # Spearman rank fit + bootstrap band
    xg2 = np.linspace(x.min(), x.max(), 200)
    slo, shi = spearman_boot_band(x, y, xg2, seed=seed + 2)
    ax.fill_between(xg2, slo, shi, color=TEAL, alpha=0.12, lw=0, zorder=2)
    xv, yv = spearman_monotone_curve(x, y)
    m = (yv >= y_lo) & (yv <= y_hi)
    ax.plot(xv[m], yv[m], color=TEAL, lw=2.4, ls=(0, (5, 3)), alpha=0.95, zorder=4)

    # orientation: best is top-right
    ax.set_ylim(y_hi, y_lo)
    ax.set_xlim((x_lo, x_hi) if higher_better else (x_hi, x_lo))
    ax.grid(True, color=GRID, lw=0.6, alpha=0.8)
    ax.tick_params(colors=INK2, labelsize=21)
    for sp in ax.spines.values():
        sp.set_edgecolor(INK); sp.set_linewidth(2.2)
    arrow = "\u2191" if higher_better else "\u2193"
    ax.set_xlabel(f"{label}{unit}  {arrow}", fontsize=27, color=INK)
    ax.set_ylabel("gFID  \u2193", fontsize=27, color=INK)

    if marginals:
        _marginal(ax_top, x, x_lo, x_hi, horizontal=False)
        _marginal(ax_right, y, y_lo, y_hi, horizontal=True)

    txt = (f"{'Pearson':<9s}r = {_s3(rp)} [{_s2(lp)}, {_s2(hp)}]\n"
           f"{'Spearman':<9s}{RHO} = {_s3(rs)} [{_s2(ls)}, {_s2(hs)}]")
    box_fs = 16 if marginals else 20
    ax.text(0.035, 0.045, txt, transform=ax.transAxes, va="bottom", ha="left",
            fontsize=box_fs, family="monospace", color=INK, zorder=5,
            bbox=dict(boxstyle="round,pad=0.4", fc=PAPER, ec=ACCENT, lw=0.9, alpha=0.97))
    nx, nha = (0.035, "left") if n_left else (0.97, "right")
    ax.text(nx, 0.965, f"n = {len(sub)}", transform=ax.transAxes,
            ha=nha, va="top", fontsize=22, color=INK)
    return present


def _half_legend(fig, anchor_x, color_keys, palette, marker_map, ncol, y=0.215):
    fam_h = [Line2D([0], [0], marker="o", linestyle="none", markersize=14,
                    markerfacecolor=palette.get(k, "#888"), markeredgecolor=PAPER, label=k)
             for k in color_keys]
    comp_h = [Line2D([0], [0], marker=mk, linestyle="none", markersize=14,
                     markerfacecolor=INK, markeredgecolor=PAPER, label=c)
              for c, mk in marker_map.items()]
    line_h = [Line2D([0], [0], color=ACCENT, lw=2.8, label="Pearson (OLS)"),
              Line2D([0], [0], color=TEAL, lw=2.6, ls=(0, (5, 3)), label=f"Spearman {RHO} (rank)")]
    leg = fig.legend(handles=fam_h + comp_h + line_h, loc="upper center",
                     bbox_to_anchor=(anchor_x, y), ncol=ncol, frameon=False,
                     fontsize=20, handletextpad=0.5, columnspacing=1.0)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.add_artist(leg)
    return leg


# --------------------------------------------------------------------------- #
# Full row
# --------------------------------------------------------------------------- #
def make_row(natural, medical, out_path, marginals=False, dpi=150):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 21,
        "axes.edgecolor": INK, "axes.linewidth": 2.2,
        "figure.facecolor": PAPER, "savefig.facecolor": PAPER,
    })
    fig = plt.figure(figsize=(31.0, 10.6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1],
                          left=0.045, right=0.993, top=0.915, bottom=0.33, wspace=0.16)
    ax_n = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    ax_m = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])]

    nat_present, med_present = [], []
    for ax, (key, label, unit, hb) in zip(ax_n, PANELS):
        nat_present += draw_panel(ax, natural, key, label, unit, hb,
                                  NATURAL_PALETTE, NAT_MARKER, marginals, seed=10)
    for ax, (key, label, unit, hb) in zip(ax_m, PANELS):
        med_present += draw_panel(ax, medical, key, label, unit, hb,
                                  FAMILY_PALETTE, MED_MARKER, marginals, seed=20, n_left=True)

    fig.canvas.draw()  # realise positions
    cx_nat = (ax_n[0].get_position().x0 + ax_n[1].get_position().x1) / 2
    cx_med = (ax_m[0].get_position().x0 + ax_m[1].get_position().x1) / 2
    fig.text(cx_nat, 0.965, "Natural imaging", ha="center", va="center",
             fontsize=27, fontweight="bold", color=INK)
    fig.text(cx_med, 0.965, "Medical imaging", ha="center", va="center",
             fontsize=27, fontweight="bold", color=INK)

    nat_keys = [k for k in NATURAL_PALETTE if k in nat_present]
    med_keys = [k for k in FAMILY_ORDER if k in med_present]
    _half_legend(fig, cx_nat, nat_keys, NATURAL_PALETTE, NAT_MARKER, ncol=5)
    _half_legend(fig, cx_med, med_keys, FAMILY_PALETTE, MED_MARKER, ncol=5)

    fig.savefig(out_path, dpi=dpi)
    if out_path.lower().endswith(".png"):
        fig.savefig(out_path[:-4] + ".pdf")
    plt.close(fig)
    print(f"Saved {out_path}  (natural n={len(nat_present)} models, "
          f"medical n={len(med_present)} families)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--natural", default=_paths.res("natural.csv"))
    ap.add_argument("--medical", default=_paths.res("tokenizer_results.csv"))
    ap.add_argument("--out", default=_paths.out("fig2_correlations.png"))
    ap.add_argument("--no_marginals", dest="marginals", action="store_false",
                    help="Drop the marginal distributions (on in the paper).")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    natural = load_natural(args.natural)
    medical = load_medical(args.medical)
    make_row(natural, medical, args.out, marginals=args.marginals, dpi=args.dpi)


if __name__ == "__main__":
    main()
