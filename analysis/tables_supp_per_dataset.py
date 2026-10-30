#!/usr/bin/env python3
"""Regenerate the per-dataset supplementary tables from tokenizer_results.csv.

Emits the 8 tables sourced from tokenizer_results (PSNR, SSIM, LPIPS, MSE, rFID,
gFID, codebook Usage/Entropy/Utility) in the exact LaTeX format used in
supplementary.tex: compression-stacked f-blocks, mean_{+-std} cells, bold
column-best within each block, dashes for missing, right-most cross-dataset mean
column. Writes analysis_out/tables_supp_per_dataset.tex.

  python analysis/tables_supp_per_dataset.py
"""
import re
import numpy as np
import pandas as pd

import _paths


df = pd.read_csv(_paths.res("tokenizer_results.csv"))
df = df[df.dataset != "_all_"].copy()
df["fam"] = df.model.str.split(".").str[1]
df["comp"] = df.model.str.extract(r"\.f(\d+)_")[0].map(lambda x: f"f{x}")

FAM10 = [("aekl","AEKL"),("soft_vq","SoftVQ"),("vq","VQGAN"),("simvq","SimVQ"),
         ("simple_qinco","SimpleQINCo"),("rqvae","RQVAE"),("diveq","DiVeQ"),
         ("sf_diveq","SF-DiVeQ"),("bsq","BSQ"),("lfq","LFQ")]
FAM8 = [f for f in FAM10 if f[0] not in ("aekl","soft_vq")]   # codebook: discrete only

DSETS = ["pathmnist","chestmnist","dermamnist","octmnist","pneumoniamnist",
         "retinamnist","breastmnist","bloodmnist","tissuemnist",
         "organamnist","organcmnist","organsmnist"]
DHEAD = ["Path","Chest","Derma","OCT","Pneu","Reti","Breast","Blood","Tiss",
         "OrgA","OrgC","OrgS"]
COMPS = [("f4","f = 4"),("f8","f = 8"),("f16","f = 16")]


# ---------- value formatters ----------
def _lead(x, nd):
    """nd-decimal fixed, dropping the leading zero before the dot (.823)."""
    s = f"{x:.{nd}f}"
    if s.startswith("0."):  s = s[1:]
    elif s.startswith("-0."): s = "-" + s[2:]
    return s

def f_psnr(x): return f"{x:.1f}"
def f_ssim(x): return _lead(x, 3)
def f_lpips(x): return _lead(x, 3)

def f_mse(x):
    if x >= 1e-3:
        return _lead(x, 4)
    m = f"{x:.1e}"               # e.g. '8.1e-04'
    mant, exp = m.split("e")
    return f"{mant}e{int(exp)}"  # -> '8.1e-4'

def f_fid(x):
    if x >= 100: return f"{x:.0f}"
    if x >= 10:  return f"{x:.1f}"
    return f"{x:.2f}"

def f_cb(x): return f"{x:.2f}"


# ---------- generic per-dataset table ----------
def make_table(col, fmt, std_col, arrow, fams, caption, label, dataset_level=False):
    """arrow: 'up'/'down'. std_col None for dataset-level metrics."""
    best_dir = (lambda s: s.max()) if arrow == "up" else (lambda s: s.min())
    L = []
    L.append(r"\begin{table*}[t]")
    L.append(r"  \centering")
    L.append(r"  \caption{" + caption + "}")
    L.append(r"  \label{" + label + "}")
    L.append(r"  \setlength{\tabcolsep}{3pt}")
    L.append(r"  \resizebox{\textwidth}{!}{%")
    L.append(r"  \begin{tabular}{lcccccccccccc|c}")
    L.append(r"  \toprule")
    L.append(r"  Method & " + " & ".join(DHEAD) + r" & mean \\")
    L.append(r"  \midrule")
    for bi, (comp, clabel) in enumerate(COMPS):
        if bi > 0:
            L.append(r"  \midrule")
        L.append(r"  \multicolumn{14}{l}{\textit{$" + clabel + r"$}} \\")
        sub = df[df.comp == comp]
        # per-dataset best value across the present families in this block
        best = {}
        for ds in DSETS:
            vals = []
            for fk, _ in fams:
                r = sub[(sub.fam == fk) & (sub.dataset == ds)]
                if len(r) and pd.notna(r.iloc[0][col]):
                    vals.append(float(r.iloc[0][col]))
            best[ds] = best_dir(pd.Series(vals)) if vals else None
        for fk, disp in fams:
            cells = [disp]
            rowvals = []
            for ds in DSETS:
                r = sub[(sub.fam == fk) & (sub.dataset == ds)]
                if not len(r) or pd.isna(r.iloc[0][col]):
                    cells.append("--"); continue
                m = float(r.iloc[0][col]); rowvals.append(m)
                body = fmt(m)
                if std_col is not None:
                    s = r.iloc[0][std_col]
                    s = 0.0 if pd.isna(s) else float(s)
                    body = body + r"$_{\pm " + fmt(s) + r"}$"
                if best[ds] is not None and abs(m - best[ds]) < 1e-9:
                    body = r"\textbf{" + body + "}"
                cells.append(body)
            cells.append(f_mean_col(rowvals, fmt))
            L.append("  " + " & ".join(cells) + r" \\")
    L.append(r"  \bottomrule")
    L.append(r"  \end{tabular}}")
    L.append(r"\end{table*}")
    return "\n".join(L)


def f_mean_col(vals, fmt):
    if not vals:
        return "--"
    return fmt(float(np.mean(vals)))


TABLES = [
    dict(col="psnr", fmt=f_psnr, std_col="psnr_std", arrow="up", fams=FAM10,
         caption=r"Per-dataset PSNR ($\uparrow$). Cells: $\text{mean}_{\pm\sigma}$ over test images.",
         label="tab:psnr"),
    dict(col="ssim", fmt=f_ssim, std_col="ssim_std", arrow="up", fams=FAM10,
         caption=r"Per-dataset SSIM ($\uparrow$). Cells: $\text{mean}_{\pm\sigma}$ over test images.",
         label="tab:ssim"),
    dict(col="lpips", fmt=f_lpips, std_col="lpips_std", arrow="down", fams=FAM10,
         caption=r"Per-dataset LPIPS ($\downarrow$). Cells: $\text{mean}_{\pm\sigma}$ over test images.",
         label="tab:lpips"),
    dict(col="mse", fmt=f_mse, std_col="mse_std", arrow="down", fams=FAM10,
         caption=r"Per-dataset pixel-MSE ($\downarrow$). Cells: $\text{mean}_{\pm\sigma}$ over test images.",
         label="tab:mse"),
    dict(col="rfid", fmt=f_fid, std_col=None, arrow="down", fams=FAM10,
         caption=r"Per-dataset reconstruction-FID ($\downarrow$). Dataset-level metric (no per-image variance).",
         label="tab:rfid"),
    dict(col="gfid", fmt=f_fid, std_col=None, arrow="down", fams=FAM10,
         caption=r"Per-dataset DiT-B/2 generation-FID ($\downarrow$). Dataset-level metric.",
         label="tab:gfid"),
    dict(col="codebook_usage", fmt=f_cb, std_col=None, arrow="up", fams=FAM8,
         caption=r"Per-dataset codebook \emph{Usage} ($\uparrow$).", label="tab:usage"),
    dict(col="codebook_entropy", fmt=f_cb, std_col=None, arrow="up", fams=FAM8,
         caption=r"Per-dataset codebook \emph{Entropy} ($\uparrow$).", label="tab:entropy"),
    dict(col="codebook_utility", fmt=f_cb, std_col=None, arrow="up", fams=FAM8,
         caption=r"Per-dataset codebook \emph{Utility} ($\uparrow$).", label="tab:utility"),
]

if __name__ == "__main__":
    out = []
    for t in TABLES:
        out.append(make_table(t["col"], t["fmt"], t["std_col"], t["arrow"],
                              t["fams"], t["caption"], t["label"]))
    tex = "\n\n".join(out) + "\n"
    with open(_paths.out("tables_supp_per_dataset.tex"), "w") as fh:
        fh.write(tex)
    print(tex)
