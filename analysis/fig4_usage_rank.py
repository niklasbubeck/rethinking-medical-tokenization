from __future__ import annotations
import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import _paths

FACTORS = [4, 8, 16]

# --- model order / labels (carried over from the geometry figure) ---
METHOD_ORDER = ["aekl", "soft_vq", "lfq", "bsq", "diveq", "sf_diveq",
                "simvq", "simple_qinco", "rqvae", "vq"]
METHOD_LABEL = {
    "vq": "VQGAN", "simvq": "SimVQ", "rqvae": "RQVAE", "diveq": "DiVeQ",
    "sf_diveq": "SF-DiVeQ", "simple_qinco": "SimQinco",
    "lfq": "LFQ", "bsq": "BSQ", "soft_vq": "SoftVQ", "aekl": "AEKL",
}
CATEGORY = {
    "vq": "explicit", "simvq": "explicit", "rqvae": "explicit",
    "diveq": "explicit", "sf_diveq": "explicit", "simple_qinco": "explicit",
    "lfq": "lookup-free", "bsq": "lookup-free",
    "soft_vq": "continuous", "aekl": "continuous",
}

# --- per-factor colours (dark -> light, echoing the reference figure) ---
F_COLOR = {4: "#3b3b6e", 8: "#8ba9cf", 16: "#e6d488"}
F_EDGE  = {4: "#26264a", 8: "#5a7aa6", 16: "#bda63f"}
OUTLINE = "#000000"   # black outline for dots and boxes
OFFSET  = {4: -0.26, 8: 0.0, 16: 0.26}   # within-group x shift
BOX_W   = 0.22

def probe_gap(csv_path):
    """Probe gap Delta = M(z_e) - M(z_q) per (family, f): mean/std over datasets.

    M is each dataset's primary metric (top-1, macro-AUROC for the multi-label
    ChestMNIST, quadratic-weighted kappa for the ordinal RetinaMNIST). Only
    discrete families have both a pre- (z_e) and post-quantization (z_q) probe.
    """
    p = pd.read_csv(csv_path)
    p["M"] = np.where(p.task_type == "ordinal", p.test_qwk,
                      np.where(p.task_type == "multilabel", p.test_auroc_macro, p.test_top1))
    p["method"] = p.model.str.split(".").str[1]
    p["factor"] = p.model.str.extract(r"\.f(\d+)_")[0].astype(int)
    w = p[p.variant.isin(["pre", "post"])].pivot_table(
        index=["method", "factor", "dataset"], columns="variant", values="M")
    gap = (w["pre"] - w["post"]).groupby(["method", "factor"]).agg(["mean", "std"])
    out = {}
    for (m, f), row in gap.iterrows():
        out.setdefault(m, {})[f] = (row["mean"], row["std"])
    return out


mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300,
    "font.size": 15, "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.titlesize": 17, "axes.labelsize": 16,
    "axes.linewidth": 1.4, "axes.axisbelow": True,
    "axes.grid": True, "grid.color": "#e9e9e9", "grid.linewidth": 0.7,
    "legend.frameon": False, "legend.fontsize": 14,
})


def darken(hexc, f):
    r, g, b = mcolors.to_rgb(hexc)
    return (r * f, g * f, b * f)


def load_long(csv_path):
    df = pd.read_csv(csv_path)
    keep = (df["model"].str.startswith("discrete")
            | df["model"].str.contains("soft_vq")
            | df["model"].str.contains("aekl"))
    df = df[keep]
    df = df[df["dataset"] != "_all_"].copy()
    df["method"] = df["model"].map(lambda m: m.split(".")[1])
    df["factor"] = df["model"].map(lambda m: int(re.search(r"f(\d+)", m.split(".")[2]).group(1)))
    df["d"] = df["model"].map(lambda m: int(re.search(r"_d(\d+)", m).group(1)))
    df = df[df["factor"].isin(FACTORS)]
    df["stable_rank_frac"] = df["latent_stable_rank"] / df["d"]
    return df


def vals(df, method, factor, col):
    v = df.loc[(df["method"] == method) & (df["factor"] == factor), col]
    return v.dropna().to_numpy()


def make_figure(df, out="usage_rank_grouped", include_mem=True, MEM=None):
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    n = len(methods)
    xpos = {m: i for i, m in enumerate(methods)}

    width = 1.0 * n + 1.0
    if include_mem:
        fig, (ax_mem, ax_top, ax_bot) = plt.subplots(
            3, 1, figsize=(width, 8.6), sharex=True,
            gridspec_kw={"height_ratios": [1, 1, 1.5], "hspace": 0.08})
        all_axes = (ax_mem, ax_top, ax_bot)
    else:
        ax_mem = None
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(width, 6.4), sharex=True,
            gridspec_kw={"height_ratios": [1, 1.5], "hspace": 0.08})
        all_axes = (ax_top, ax_bot)

    # group separators
    for ax in all_axes:
        for i in range(n - 1):
            ax.axvline(i + 0.5, ls=(0, (4, 4)), lw=0.8, color="#c9c9c9", zorder=0)

    rng = np.random.default_rng(0)

    # ---------- TOP: memorization (mean +/- std) as dots with error bars ----------
    if include_mem:
        ax_mem.axhline(0.0, ls="--", lw=0.9, color="#888888", zorder=1)
        for m in methods:
            if m not in MEM:
                continue
            for f in FACTORS:
                mean, std = MEM[m][f]
                x = xpos[m] + OFFSET[f]
                ax_mem.errorbar(x, mean, yerr=std, fmt="o", ms=8,
                                mfc=F_COLOR[f], mec=OUTLINE, mew=1.1,
                                ecolor=OUTLINE, elinewidth=1.0, capsize=3,
                                capthick=1.0, zorder=4)
        ax_mem.set_ylabel("Probing gap \u0394\u200a\u2193")
        for m in methods:
            if CATEGORY.get(m) == "continuous" and vals(df, m, FACTORS[0], "codebook_usage").size == 0:
                ax_mem.text(xpos[m], 0.5, "No codebook",
                            transform=ax_mem.get_xaxis_transform(),
                            ha="center", va="center", rotation=90,
                            fontsize=12, color="#999999", style="italic")
    for m in methods:
        x0 = xpos[m]
        for f in FACTORS:
            v = vals(df, m, f, "codebook_usage")
            if v.size == 0:
                continue
            x = x0 + OFFSET[f]
            jit = rng.uniform(-0.07, 0.07, v.size)
            ax_top.scatter(x + jit, v, s=26, color=F_COLOR[f], alpha=0.26,
                           linewidths=0, zorder=2)
            ax_top.scatter([x], [v.mean()], s=70, color=F_COLOR[f],
                           edgecolor=OUTLINE, linewidth=1.3, zorder=4)
    # mark codebook-free models
    for m in methods:
        if CATEGORY.get(m) == "continuous" and vals(df, m, FACTORS[0], "codebook_usage").size == 0:
            ax_top.annotate("No codebook", xy=(xpos[m], 0.5), ha="center",
                            va="center", rotation=90, fontsize=12,
                            color="#999999", style="italic")
    ax_top.set_ylabel("Codebook usage\u200a\u2191")
    ax_top.set_ylim(-0.04, 1.06)

    # ---------- BOTTOM: stable rank as boxplots ----------
    for f in FACTORS:
        data, positions = [], []
        for m in methods:
            v = vals(df, m, f, "stable_rank_frac")
            if v.size == 0:
                continue
            data.append(v)
            positions.append(xpos[m] + OFFSET[f])
        if not data:
            continue
        bp = ax_bot.boxplot(data, positions=positions, widths=BOX_W,
                            patch_artist=True, showfliers=False, zorder=3,
                            medianprops=dict(color=OUTLINE, lw=1.6),
                            whiskerprops=dict(color=OUTLINE, lw=1.1),
                            capprops=dict(color=OUTLINE, lw=1.1))
        for box in bp["boxes"]:
            box.set_facecolor(mcolors.to_rgba(F_COLOR[f], 0.6))
            box.set_edgecolor(OUTLINE)
            box.set_linewidth(1.4)
        # faint individual datasets on top of each box
        for m in methods:
            v = vals(df, m, f, "stable_rank_frac")
            if v.size == 0:
                continue
            x = xpos[m] + OFFSET[f]
            jit = rng.uniform(-0.06, 0.06, v.size)
            ax_bot.scatter(x + jit, v, s=20, color=F_EDGE[f], alpha=0.30,
                           linewidths=0, zorder=4)
    ax_bot.set_ylabel("Stable rank / d\u200a\u2191")
    ax_bot.set_ylim(0, 1.03)

    # x axis
    ax_bot.set_xticks(range(n))
    ax_bot.set_xticklabels([METHOD_LABEL[m] for m in methods],
                           rotation=30, ha="right")
    ax_bot.set_xlim(-0.6, n - 0.4)
    for ax in all_axes:
        ax.grid(axis="x", visible=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(True)

    # legend (compression factors) in the top-right corner of the top panel
    top_axis = ax_mem if include_mem else ax_top
    handles = [Patch(facecolor=mcolors.to_rgba(F_COLOR[f], 0.6),
                     edgecolor=OUTLINE, label=f"f{f}") for f in FACTORS]
    top_axis.legend(handles=handles, title="Compression factor",
                    loc="upper right", ncol=3, frameon=True, framealpha=0.9,
                    edgecolor="#cccccc", facecolor="white",
                    columnspacing=1.0, handletextpad=0.5,
                    fontsize=14, title_fontsize=14)

    fig.align_ylabels(all_axes)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight",
                    transparent=False, facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    d = load_long(_paths.res("tokenizer_results.csv"))
    gaps = probe_gap(_paths.res("linear_probing.csv"))
    make_figure(d, out=_paths.out("fig4_usage_rank"), include_mem=True, MEM=gaps)
