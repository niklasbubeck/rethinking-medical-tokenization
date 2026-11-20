#!/usr/bin/env python3
"""Export the paper's checkpoints as weights-only safetensors for the GitHub release.

Maintainer script. For each of the 30 configurations it takes the highest-numbered
training checkpoint (the one every evaluation script loads) of

  tokenizer  <src>/<model>/train/models/ckpt.<N>.pt      -> tokenizer.<model>.safetensors
  generator  <src>/dit_<model>/train/models/ckpt.<N>.pt  -> dit.<model>.safetensors

and keeps only `state["model"]` (optimizer, discriminator and RNG state are dropped).
The tokenizer file is the frozen first stage the DiT was trained on (see
`export_pair`); the DiT file holds the rest of the generation wrapper (DiT-B/2 and
the latent scale factor). tools/download_checkpoints.py joins them again. Weights
are stored in their original fp32. A manifest with sizes and SHA-256 checksums is
written to tools/checkpoints.json.

  python tools/export_checkpoints.py --src /path/to/outputs/medtok --out ./release_weights
"""
import argparse
import glob
import hashlib
import json
import os
import re

import torch
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "tools", "checkpoints.json")
REPO = "niklasbubeck/rethinking-medical-tokenization"
TAG = "weights-v1"

FAMILIES = ["continuous.aekl", "continuous.soft_vq", "discrete.vq", "discrete.simvq",
            "discrete.simple_qinco", "discrete.rqvae", "discrete.diveq", "discrete.sf_diveq",
            "discrete.lfq", "discrete.bsq"]
SHAPES = {
    "continuous.aekl": ["f4_d3", "f8_d4", "f16_d8"],
    "lookup_free": ["f4_d10_b10", "f8_d14_b14", "f16_d18_b18"],
    "codebook": ["f4_d3_e8192", "f8_d4_e16384", "f16_d8_e16384"],
}
MODELS = [f"{fam}.{s}" for fam in FAMILIES for s in SHAPES[
    "continuous.aekl" if fam == "continuous.aekl"
    else "lookup_free" if fam in ("discrete.lfq", "discrete.bsq") else "codebook"]]


def latest(run_dir):
    ckpts = [p for p in glob.glob(os.path.join(run_dir, "train", "models", "ckpt.*.pt"))
             if re.search(r"ckpt\.\d+\.pt$", p)]
    if not ckpts:
        raise FileNotFoundError(f"no ckpt.<N>.pt under {run_dir}/train/models")
    return max(ckpts, key=lambda p: int(re.search(r"ckpt\.(\d+)\.pt$", p).group(1)))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def strip(sd):
    return {(k[len("module."):] if k.startswith("module.") else k): v.detach().contiguous().clone()
            for k, v in sd.items()}


def save(sd, dst, meta):
    save_file(sd, dst, metadata={k: str(v) for k, v in meta.items()})
    return {"file": os.path.basename(dst), **meta,
            "bytes": os.path.getsize(dst), "sha256": sha256(dst)}


def export_pair(src, out, model):
    """Export the DiT and the tokenizer it was trained with.

    The released tokenizer is the first stage stored inside the DiT checkpoint, so
    tokenizer and generator always match. For 24 configurations it is identical to
    the standalone tokenizer checkpoint; for LFQ and BSQ the DiTs were trained on a
    second compute cluster against that cluster's tokenizer run, which is released.
    """
    tok_ckpt, dit_ckpt = latest(os.path.join(src, model)), latest(os.path.join(src, f"dit_{model}"))
    num = lambda p: int(re.search(r"ckpt\.(\d+)\.pt$", p).group(1))
    dit = torch.load(dit_ckpt, map_location="cpu", weights_only=False)
    dsd = strip(dit["model"])
    first = {k[len("first_stage."):]: v for k, v in dsd.items() if k.startswith("first_stage.")}
    tok = torch.load(tok_ckpt, map_location="cpu", weights_only=False)
    tsd = strip(tok["model"])
    same = tsd.keys() == first.keys() and all(torch.equal(tsd[k], first[k]) for k in tsd)
    tmeta = ({"model": model, "kind": "tokenizer", "ckpt": num(tok_ckpt), "steps": tok.get("steps"),
              "source": "standalone tokenizer run"} if same else
             {"model": model, "kind": "tokenizer", "ckpt": num(dit_ckpt), "steps": "",
              "source": "first stage of the DiT checkpoint (tokenizer run the DiT was trained on)"})
    entry = {"tokenizer": save(tsd if same else first, os.path.join(out, f"tokenizer.{model}.safetensors"), tmeta)}
    del tok, tsd
    gen = {k: v for k, v in dsd.items() if not k.startswith("first_stage.")}
    entry["dit"] = save(gen, os.path.join(out, f"dit.{model}.safetensors"),
                        {"model": model, "kind": "dit", "ckpt": num(dit_ckpt), "steps": dit.get("steps"),
                         "source": "DiT-B/2 and latent scale factor; first stage = tokenizer file"})
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="folder holding <model>/ and dit_<model>/ runs")
    ap.add_argument("--out", required=True, help="output folder for the .safetensors files")
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    manifest = {"repo": REPO, "tag": TAG, "models": {}}
    if os.path.isfile(MANIFEST):
        manifest = json.load(open(MANIFEST))
    for model in args.models:
        entry = export_pair(args.src, args.out, model)
        for kind, e in entry.items():
            print(f"{model:38s} {kind:9s} {e['file']} ({e['bytes'] / 1e6:.0f} MB; {e['source']})", flush=True)
        manifest["models"][model] = entry
        with open(MANIFEST, "w") as f:
            json.dump(manifest, f, indent=2)
    print(f"manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
