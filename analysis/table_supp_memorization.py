#!/usr/bin/env python3
"""Regenerate a memorization supplementary table from memorization.csv.

Usage: python analysis/table_supp_memorization.py --metric copy_rate --extractor sscd
  metric    : copy_rate (lower=better, bold min) | mem_ratio (higher=better, bold max)
  extractor : sscd (primary) | dinov2 (validation)

copy_rate = fraction of generated samples above the calibrated test->train
similarity floor (memorization above natural self-similarity). mem_ratio =
median Somepalli ratio d1/mean(d2..d_{k+1}); lower => tighter copy of one
training image, so higher mem_ratio is better. Writes the LaTeX table to
analysis_out/ and prints it.
"""
import argparse
import numpy as np
import pandas as pd

import _paths

_ap = argparse.ArgumentParser()
_ap.add_argument("--csv", default=_paths.res("memorization.csv"))
_ap.add_argument("--metric", choices=["copy_rate", "mem_ratio"], default="copy_rate")
_ap.add_argument("--extractor", choices=["sscd", "dinov2"], default="sscd")
_args = _ap.parse_args()
CSV, METRIC, EXTR = _args.csv, _args.metric, _args.extractor
LOWER_BETTER = (METRIC == "copy_rate")

df = pd.read_csv(CSV)
df = df[df.extractor == EXTR]
df["fam"] = df.model.str.split(".").str[1]
df["comp"] = df.model.str.extract(r"\.f(\d+)_")[0].map(lambda x: f"f{x}")

FAM10 = [("aekl", "AEKL"), ("soft_vq", "SoftVQ"), ("vq", "VQGAN"),
         ("simvq", "SimVQ"), ("simple_qinco", "SimpleQINCo"), ("rqvae", "RQVAE"),
         ("diveq", "DiVeQ"), ("sf_diveq", "SF-DiVeQ"), ("bsq", "BSQ"), ("lfq", "LFQ")]
DSETS = ["pathmnist", "chestmnist", "dermamnist", "octmnist", "pneumoniamnist",
         "retinamnist", "breastmnist", "bloodmnist", "tissuemnist",
         "organamnist", "organcmnist", "organsmnist"]
DHEAD = ["Path", "Chest", "Derma", "OCT", "Pneu", "Reti", "Breast", "Blood",
         "Tiss", "OrgA", "OrgC", "OrgS"]
COMPS = [("f4", "f = 4"), ("f8", "f = 8"), ("f16", "f = 16")]


def _lead(x, nd=3):
    s = f"{x:.{nd}f}"
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return s


val = {}
for (fam, comp, ds), sub in df.groupby(["fam", "comp", "dataset"]):
    v = sub.iloc[0][METRIC]
    if pd.notna(v):
        val[(fam, comp, ds)] = float(v)

_MNAME = {"copy_rate": r"copy rate", "mem_ratio": r"memorization ratio"}[METRIC]
_ARROW = r"$\downarrow$" if LOWER_BETTER else r"$\uparrow$"
_EXTRDISP = {"sscd": "SSCD", "dinov2": "DINOv2"}[EXTR]
_LABEL = f"tab:mem-{METRIC.replace('_','-')}-{EXTR}"

L = []
L.append(r"\begin{table*}[t]")
L.append(r"  \centering")
L.append(r"  \caption{Per-dataset " + _MNAME + r" (" + _ARROW +
         r"), " + _EXTRDISP + r" features.}")
L.append(r"  \label{" + _LABEL + "}")
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
    best = {}
    for ds in DSETS:
        vals = [val[(fk, comp, ds)] for fk, _ in FAM10 if (fk, comp, ds) in val]
        if vals:
            best[ds] = (min if LOWER_BETTER else max)(vals)
        else:
            best[ds] = None
    for fk, disp in FAM10:
        cells = [disp]
        rowvals = []
        for ds in DSETS:
            if (fk, comp, ds) not in val:
                cells.append("--")
                continue
            m = val[(fk, comp, ds)]
            rowvals.append(m)
            body = _lead(m, 3)
            if best[ds] is not None and abs(m - best[ds]) < 1e-9:
                body = r"\textbf{" + body + "}"
            cells.append(body)
        cells.append(_lead(float(np.mean(rowvals)), 3) if rowvals else "--")
        L.append("  " + " & ".join(cells) + r" \\")
L.append(r"  \bottomrule")
L.append(r"  \end{tabular}}")
L.append(r"\end{table*}")
tex = "\n".join(L)
open(_paths.out(f"table_supp_memorization_{METRIC}_{EXTR}.tex"), "w").write(tex + "\n")
print(tex)
