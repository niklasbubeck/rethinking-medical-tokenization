#!/usr/bin/env python3
"""Build results/tokenizer_results.csv from the pipeline outputs.

Concatenates every run's `evaluate_tokenizer.py` output
(<RUNS>/<model>/evaluation/results.csv: reconstruction, codebook and latent
metrics, one row per dataset plus a pooled `_all_` row) and merges the
Frechet distances computed offline by `calculate_fids.py`:

  rFID / rFMD  <- --recon_fids  (calculate_fids.py --mode reconstruction)
  gFID / gFMD  <- --gen_fids    (calculate_fids.py --mode generation)

`calculate_fids.py` appends to its CSV, so when a configuration was scored more
than once the most recent row wins.

  python analysis/aggregate_results.py \
      --recon_fids fids_reconstruction.csv --gen_fids fids_generation.csv
"""
import argparse
import os

import pandas as pd

import _paths

# The 30 configurations of the paper: 10 families x f in {4, 8, 16}.
ROSTER = [f"{fam}.{cfg}" for fam, cfgs in [
    ("continuous.aekl", ["f4_d3", "f8_d4", "f16_d8"]),
    ("continuous.soft_vq", ["f4_d3_e8192", "f8_d4_e16384", "f16_d8_e16384"]),
    ("discrete.vq", ["f4_d3_e8192", "f8_d4_e16384", "f16_d8_e16384"]),
    ("discrete.simvq", ["f4_d3_e8192", "f8_d4_e16384", "f16_d8_e16384"]),
    ("discrete.simple_qinco", ["f4_d3_e8192", "f8_d4_e16384", "f16_d8_e16384"]),
    ("discrete.rqvae", ["f4_d3_e8192", "f8_d4_e16384", "f16_d8_e16384"]),
    ("discrete.diveq", ["f4_d3_e8192", "f8_d4_e16384", "f16_d8_e16384"]),
    ("discrete.sf_diveq", ["f4_d3_e8192", "f8_d4_e16384", "f16_d8_e16384"]),
    ("discrete.bsq", ["f4_d10_b10", "f8_d14_b14", "f16_d18_b18"]),
    ("discrete.lfq", ["f4_d10_b10", "f8_d14_b14", "f16_d18_b18"]),
] for cfg in cfgs]


def load_fids(path, prefix):
    """(model, dataset) -> {prefix}fid, {prefix}fmd, keeping the latest row."""
    df = pd.read_csv(path).drop_duplicates(["model", "dataset"], keep="last")
    return df[["model", "dataset", "fid", "fmd"]].rename(
        columns={"fid": f"{prefix}fid", "fmd": f"{prefix}fmd"})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", default=_paths.RUNS, help="<base_dir>/outputs/<project>/")
    ap.add_argument("--recon_fids", required=True)
    ap.add_argument("--gen_fids", required=True)
    ap.add_argument("--models", nargs="+", default=ROSTER)
    ap.add_argument("--out", default=_paths.out("tokenizer_results.csv"))
    a = ap.parse_args()

    frames, missing = [], []
    for m in a.models:
        p = os.path.join(a.runs, m, "evaluation", "results.csv")
        if not os.path.isfile(p):
            missing.append(m)
            continue
        df = pd.read_csv(p)
        df.insert(0, "model", m)
        frames.append(df)
    if not frames:
        raise SystemExit(f"no evaluation/results.csv found under {a.runs}")
    tok = pd.concat(frames, ignore_index=True)
    tok = tok.drop(columns=[c for c in ("rfid", "rfmd", "gfid", "gfmd") if c in tok.columns])
    tok = tok.merge(load_fids(a.recon_fids, "r"), on=["model", "dataset"], how="left")
    tok = tok.merge(load_fids(a.gen_fids, "g"), on=["model", "dataset"], how="left")

    # Place the Frechet distances right after the pixel metrics.
    cols = [c for c in tok.columns if c not in ("rfid", "rfmd", "gfid", "gfmd")]
    at = cols.index("mae_std") + 1 if "mae_std" in cols else len(cols)
    tok = tok[cols[:at] + ["rfid", "rfmd", "gfid", "gfmd"] + cols[at:]]

    tok.to_csv(a.out, index=False)
    per_ds = tok[tok.dataset != "_all_"]
    print(f"models: {len(frames)}  missing: {missing or 'none'}")
    print(f"per-dataset rows: {len(per_ds)}  with gFID: {per_ds.gfid.notna().sum()}  "
          f"with rFID: {per_ds.rfid.notna().sum()}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
