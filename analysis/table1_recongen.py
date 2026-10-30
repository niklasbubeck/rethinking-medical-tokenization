#!/usr/bin/env python3
"""Table 1: holistic reconstruction + generation overview.
Compression factors as multicolumn groups, 10 families as rows, 6 metrics each.
  PSNR, SSIM : mean over datasets (tokenizer_results.csv)
  rFID       : mean over datasets   (fids_reconstruction.csv: fid)
  rFMD       : median over datasets  (fids_reconstruction.csv: fmd)
  gFID       : mean over datasets   (tokenizer_results.csv: gfid)
  gFMD       : median over datasets  (tokenizer_results.csv: gfmd)
Only the central value is shown (spreads -> supplementary).

  python analysis/table1_recongen.py"""
import os, re
import numpy as np
import pandas as pd

import _paths

TOK = _paths.res("tokenizer_results.csv")
REC = _paths.res("fids_reconstruction.csv")

DISPLAY = {"aekl":"AEKL","soft_vq":"SoftVQ","vq":"VQGAN","simvq":"SimVQ",
           "simple_qinco":"SimQINCo","rqvae":"RQVAE","diveq":"DiVeQ",
           "sf_diveq":"SF-DiVeQ","bsq":"BSQ","lfq":"LFQ"}
FAMILIES = ["AEKL","SoftVQ","VQGAN","SimVQ","SimQINCo","RQVAE",
            "DiVeQ","SF-DiVeQ","BSQ","LFQ"]
COMPS = ["f4","f8","f16"]

def fam_comp(model):
    fam = DISPLAY.get(model.split(".")[1], model.split(".")[1])
    comp = "f" + re.search(r"f(\d+)", model).group(1)
    return fam, comp

# ---- load + aggregate real data ----
tok = pd.read_csv(TOK)
tok = tok[tok.dataset != "_all_"]                      # drop pooled codebook row
tokg = tok.groupby("model").agg(PSNR=("psnr","mean"), SSIM=("ssim","mean"))
rec = pd.read_csv(REC)
recg = rec.groupby("model").agg(rFID=("fid","mean"), rFMD=("fmd","median"))

# gFID/gFMD: from merged tokenizer_results.csv if present, else 0 placeholder
have_gen = "gfid" in tok.columns and tok["gfid"].notna().any()
if have_gen:
    geng = tok.groupby("model").agg(gFID=("gfid","mean"), gFMD=("gfmd","median"))
else:
    geng = None

cells_data = {}
for model, r in tokg.iterrows():
    cells_data.setdefault(fam_comp(model), {}).update(PSNR=r.PSNR, SSIM=r.SSIM)
for model, r in recg.iterrows():
    cells_data.setdefault(fam_comp(model), {}).update(rFID=r.rFID, rFMD=r.rFMD)
if geng is not None:
    for model, r in geng.iterrows():
        cells_data.setdefault(fam_comp(model), {}).update(gFID=r.gFID, gFMD=r.gFMD)

# label, arrow, key, fmt
METRICS = [
    ("PSNR", "up",   "PSNR", "{:.1f}"),
    ("SSIM", "up",   "SSIM", "{:.3f}"),
    ("rFID", "down", "rFID", "{:.1f}"),
    ("rFMD", "down", "rFMD", "{:.2f}"),
    ("gFID", "down", "gFID", "{:.1f}"),
    ("gFMD", "down", "gFMD", "{:.1f}"),
]
ARROW = {"up": r"$\uparrow$", "down": r"$\downarrow$"}
NM = len(METRICS)
DIRN = {key: d for _, d, key, _ in METRICS}
DEC = {key: int(re.search(r"\.(\d+)f", fmt).group(1)) for _, _, key, fmt in METRICS}

def cell_val(fam, comp, key):
    if key in ("gFID", "gFMD") and not have_gen:
        return 0.0
    v = cells_data.get((fam, comp), {}).get(key)
    return None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

# best/second-best per (metric,comp), comparing the displayed (rounded) values
RANK = {}
for _, _, key, fmt in METRICS:
    for c in COMPS:
        vals = {f: round(cell_val(f, c, key), DEC[key]) for f in FAMILIES if cell_val(f, c, key) is not None}
        distinct = sorted(set(vals.values()))
        if len(distinct) < 2:                       # placeholder/constant column -> no marking
            RANK[(key, c)] = (None, None); continue
        best, second = (distinct[-1], distinct[-2]) if DIRN[key] == "up" else (distinct[0], distinct[1])
        RANK[(key, c)] = (best, second)

def fmt_cell(fam, comp, key, fmt):
    v = cell_val(fam, comp, key)
    if v is None:
        return "--"
    s = fmt.format(v)
    best, second = RANK.get((key, comp), (None, None))
    rv = round(v, DEC[key])
    if best is not None and rv == best:
        s = r"\textbf{" + s + "}"
    elif second is not None and rv == second:
        s = r"\underline{" + s + "}"
    return s

spec = "l" + "".join("c"*NM for _ in COMPS)
L = []
gen_status = "real data" if have_gen else "0 (placeholder); regenerate once generation FID/FMD is computed"
L.append(f"% gFID/gFMD = {gen_status}")
L.append(r"\begin{table*}[t]")
L.append(r"  \centering")
L.append(r"  \caption{\textbf{Holistic reconstruction and generation overview.} "
         r"Cross-dataset central tendency over the 12 MedMNIST+ datasets, grouped "
         r"by compression $f$. Per-dataset values vary strongly: dataset difficulty "
         r"differs, and the number of images entering each FID/FMD estimate ranges "
         r"from a few hundred (e.g.\ BreastMNIST) to tens of thousands "
         r"(TissueMNIST), to which Fr\'echet distances are sensitive. Cells are "
         r"therefore the cross-dataset mean, except rFMD/gFMD which report the more "
         r"robust \emph{median} (each dataset also uses its own MedMNIST feature "
         r"extractor). Per-dataset values and spreads are in the supplementary. "
         r"$\uparrow$/$\downarrow$ indicate better direction; FMD~$=$~Fr\'echet "
         r"MedMNIST Distance."
         + ("" if have_gen else r" \emph{(gFID/gFMD pending generation.)}}")
         + ("}" if have_gen else ""))
L.append(r"  \label{tab:recongen}")
L.append(r"  \setlength{\tabcolsep}{4pt}")
L.append(r"  \resizebox{\textwidth}{!}{%")
L.append(r"  \begin{tabular}{" + spec + "}")
L.append(r"  \toprule")
hdr, cmids, start = [], [], 2
for c in COMPS:
    hdr.append(r"\multicolumn{%d}{c}{$f{=}%s$}" % (NM, c[1:]))
    cmids.append(r"\cmidrule(lr){%d-%d}" % (start, start+NM-1)); start += NM
L.append(r"  & " + " & ".join(hdr) + r" \\")
L.append("  " + "".join(cmids))
sub = ["Method"] + [lab + ARROW[ar] for lab,ar,_,_ in METRICS]*len(COMPS)
L.append("  " + " & ".join(sub) + r" \\")
L.append(r"  \midrule")
for fam in FAMILIES:
    cells = [fam]
    for c in COMPS:
        for _, _, key, fmt in METRICS:
            cells.append(fmt_cell(fam, c, key, fmt))
    L.append("  " + " & ".join(cells) + r" \\")
L.append(r"  \bottomrule")
L.append(r"  \end{tabular}%")
L.append(r"  }")
L.append(r"\end{table*}")

tex = "\n".join(L) + "\n"
open(_paths.out("table1_recongen.tex"), "w").write(tex)
print(tex)
