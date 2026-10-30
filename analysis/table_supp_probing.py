#!/usr/bin/env python3
"""Regenerate the Linear Probing supplementary table (tab:probe) from a
linear_probing.csv. Probes the frozen PRE-quantization latent z_e, so for
discrete models we select variant=='pre' (continuous models use 'default').
Per-dataset primary metric: top-1 (multiclass), macro-AUROC (multilabel
ChestMNIST), quadratic-weighted kappa (ordinal RetinaMNIST). Prints the LaTeX
table block to analysis_out/ and stdout.

  python analysis/table_supp_probing.py            # z_e (paper)
  python analysis/table_supp_probing.py --latent zq
"""
import argparse
import numpy as np
import pandas as pd

import _paths

_ap = argparse.ArgumentParser()
_ap.add_argument("--csv", default=_paths.res("linear_probing.csv"))
# latent: "ze" (pre-quantization) or "zq" (post-quantization)
_ap.add_argument("--latent", choices=["ze", "zq"], default="ze")
_args = _ap.parse_args()
CSV, LATENT = _args.csv, _args.latent
df = pd.read_csv(CSV)
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

def pick_variant(sub):
    # ze probes the pre-quantization latent (continuous models use 'default');
    # zq probes the post-quantization latent (only discrete models have it).
    order = ["pre", "default"] if LATENT == "ze" else ["post"]
    for v in order:
        s = sub[sub.variant == v]
        if len(s):
            return s.iloc[0]
    return None


def _lead(x, nd=3):
    s = f"{x:.{nd}f}"
    if s.startswith("0."):
        s = s[1:]
    elif s.startswith("-0."):
        s = "-" + s[2:]
    return s


# value lookup keyed by (fam, comp, dataset)
val = {}
for (fam, comp, ds), sub in df.groupby(["fam", "comp", "dataset"]):
    row = pick_variant(sub)
    if row is None:
        continue
    v = row["val_value"]   # validation primary metric (AUROC-macro / QWK for retina)
    if pd.notna(v):
        val[(fam, comp, ds)] = float(v)

L = []
L.append(r"\begin{table*}[t]")
L.append(r"  \centering")
_LAT = r"z_e" if LATENT == "ze" else r"z_q"
_LABEL = "tab:probe" if LATENT == "ze" else "tab:probe-zq"
L.append(r"  \caption{Per-dataset linear-probing primary metric ($\uparrow$, on $" + _LAT + r"$).}")
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
        best[ds] = max(vals) if vals else None
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
open(_paths.out(f"table_supp_probing_{LATENT}.tex"), "w").write(tex + "\n")
print(tex)
