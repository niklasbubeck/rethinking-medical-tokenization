#!/usr/bin/env python3
"""Collect the rate-distortion summary at one compression factor.

Reads the pooled cell of each discrete family's codebook_misalignment.py output
(<RUNS>/<model>/evaluation/codebook_alignment/results.json) and writes one row
per family: allocated bits B, empirical code entropy H(C) = sum_l H(C_l),
quantization error D, D/B, D/H(C), codebook usage and the per-level entropies.
The released copy is results/dperbit_entropy_f8.csv.

  python analysis/collect_rate_distortion.py --f 8
"""
import argparse
import csv
import json
import os

import _paths

# display name -> model key at each compression factor
FAMILIES = [("VQGAN", "discrete.vq"), ("SimVQ", "discrete.simvq"),
            ("SimQINCo", "discrete.simple_qinco"), ("RQVAE", "discrete.rqvae"),
            ("DiVeQ", "discrete.diveq"), ("SF-DiVeQ", "discrete.sf_diveq"),
            ("LFQ", "discrete.lfq"), ("BSQ", "discrete.bsq")]
SUFFIX = {"discrete.lfq": {4: "f4_d10_b10", 8: "f8_d14_b14", 16: "f16_d18_b18"},
          "discrete.bsq": {4: "f4_d10_b10", 8: "f8_d14_b14", 16: "f16_d18_b18"}}
DEFAULT_SUFFIX = {4: "f4_d3_e8192", 8: "f8_d4_e16384", 16: "f16_d8_e16384"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--f", type=int, default=8, choices=[4, 8, 16])
    ap.add_argument("--runs", default=_paths.RUNS)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or _paths.out(f"dperbit_entropy_f{a.f}.csv")

    rows = []
    for name, key in FAMILIES:
        model = f"{key}.{SUFFIX.get(key, DEFAULT_SUFFIX)[a.f]}"
        path = os.path.join(a.runs, model, "evaluation", "codebook_alignment", "results.json")
        if not os.path.isfile(path):
            print(f"[skip] {model}: no {path}")
            continue
        r = json.load(open(path))
        cell = next(c for c in r["cells"] if c["dataset"] == "pooled")
        rows.append([name, r["bits_per_token"], round(cell["entropy_bits"], 3), cell["distortion"],
                     cell["distortion_per_bit"], cell["distortion_per_bit_entropy"],
                     round(100 * cell["codebook_utilization"], 1),
                     ";".join(f"{h:.2f}" for h in cell["per_level_entropy_bits"])])
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "B_bits", "H_C_bits", "D", "D_per_B_old", "D_per_H_new",
                    "usage_pct", "per_level_H"])
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} families)")


if __name__ == "__main__":
    main()
