#!/usr/bin/env python3
"""Reconstruction <-> generation correlation tables (supplementary).

For rFID and PSNR against DiT-B/2 gFID, reports Pearson r and Spearman rho with
95% percentile-bootstrap CIs (B=10,000 resamples, seed 0) per compression factor
and pooled over all (tokenizer, dataset) pairs, plus a cluster-robust bootstrap
that resamples whole datasets so the repeated tokenizer measurements within a
dataset are not treated as independent.

  python analysis/tables_correlation.py
"""
import re

import numpy as np
import pandas as pd
from scipy import stats

import _paths

B, SEED = 10_000, 0
METRICS = [("rFID", "rFID correlates positively with gFID at every compression, "
                    "and the association strengthens as the bottleneck tightens."),
           ("PSNR", "PSNR correlates negatively (higher PSNR $\\rightarrow$ lower/better "
                    "gFID) at every compression.")]


def load(csv_path):
    """One row per (tokenizer, dataset) with a generation FID."""
    df = pd.read_csv(csv_path)
    df = df[~df.dataset.str.contains("all")].copy()
    df["compression"] = df.model.map(lambda m: "f" + re.search(r"\.f(\d+)_", m).group(1))
    df = df.rename(columns={"rfid": "rFID", "psnr": "PSNR", "gfid": "gFID"})
    return df[df.gFID.notna() & np.isfinite(df.gFID)].reset_index(drop=True)


def bootstrap_ci(x, y, kind="pearson"):
    """95% percentile bootstrap CI, vectorised over the B resamples.

    Spearman is the Pearson correlation of within-resample ranks (ties are
    negligible for these continuous metrics).
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    idx = np.random.default_rng(SEED).integers(0, len(x), size=(B, len(x)))
    xb, yb = x[idx], y[idx]
    if kind == "spearman":
        xb = xb.argsort(1).argsort(1).astype(float)
        yb = yb.argsort(1).argsort(1).astype(float)
    xm, ym = xb - xb.mean(1, keepdims=True), yb - yb.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (xm * ym).sum(1) / np.sqrt((xm ** 2).sum(1) * (ym ** 2).sum(1))
    return np.nanpercentile(r, [2.5, 97.5])


def clustered_ci(df, col):
    """Pearson r with a bootstrap over whole datasets (clusters)."""
    groups = [g for _, g in df.groupby("dataset")]
    rng = np.random.default_rng(SEED)
    rs = []
    for _ in range(B):
        cat = pd.concat([groups[i] for i in rng.integers(0, len(groups), size=len(groups))])
        rs.append(stats.pearsonr(cat[col], cat.gFID)[0])
    return stats.pearsonr(df[col], df.gFID)[0], *np.nanpercentile(rs, [2.5, 97.5])


def stat_row(sub, col):
    sub = sub[sub[col].notna() & np.isfinite(sub[col])]
    x, y = sub[col].to_numpy(float), sub.gFID.to_numpy(float)
    return dict(n=len(x), rp=stats.pearsonr(x, y)[0], rp_ci=bootstrap_ci(x, y, "pearson"),
                rs=stats.spearmanr(x, y)[0], rs_ci=bootstrap_ci(x, y, "spearman"))


def fmt(v, ci, bold=False):
    s = lambda t: f"{t:+.2f}"
    val = f"\\mathbf{{{s(v)}}}" if bold else s(v)
    return f"${val}$ $[{s(ci[0])}, {s(ci[1])}]$"


def log(col, label, s):
    print(f"  {col:5s} {label:7s} n={s['n']:3d}  r={s['rp']:+.2f} [{s['rp_ci'][0]:+.2f}, {s['rp_ci'][1]:+.2f}]"
          f"  rho={s['rs']:+.2f} [{s['rs_ci'][0]:+.2f}, {s['rs_ci'][1]:+.2f}]")


def table(df, col, caption_tail):
    lines = [r"\begin{table}[t]", r"\centering",
             f"\\caption{{Per-compression correlation of \\textbf{{{col}}} with DiT-B/2 "
             f"generation-FID, with 95\\% bootstrap CIs ($B{{=}}10{{,}}000$). {caption_tail}}}",
             f"\\label{{tab:corr-{col.lower()}-comp}}",
             r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lcc}", r"\toprule",
             r"Compression & Pearson $r$ [95\% CI] & Spearman $\rho$ [95\% CI] \\", r"\midrule"]
    for f in ("f4", "f8", "f16"):
        s = stat_row(df[df.compression == f], col)
        log(col, f, s)
        lines.append(f"$f{{={f[1:]}}}$ & {fmt(s['rp'], s['rp_ci'])} & {fmt(s['rs'], s['rs_ci'])} \\\\")
    s = stat_row(df, col)
    log(col, "pooled", s)
    rc, lo, hi = clustered_ci(df, col)
    print(f"  {col:5s} cluster-robust (by dataset)  r={rc:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    lines += [r"\midrule",
              f"\\textbf{{Pooled}} & {fmt(s['rp'], s['rp_ci'], True)} & {fmt(s['rs'], s['rs_ci'], True)} \\\\",
              r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    return "\n".join(lines)


def main():
    df = load(_paths.res("tokenizer_results.csv"))
    print(f"{len(df)} (tokenizer, dataset) pairs with gFID")
    tex = "\n".join(table(df, col, cap) for col, cap in METRICS)
    out = _paths.out("tables_correlation.tex")
    open(out, "w").write(tex)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
