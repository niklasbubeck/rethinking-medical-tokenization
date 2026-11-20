#!/usr/bin/env python3
"""Download the paper's pretrained tokenizers and DiT-B/2 generators.

The weights are attached to the GitHub release listed in tools/checkpoints.json.
Each file is verified against its SHA-256 checksum and then written in the layout
the training and evaluation scripts expect, so they run unchanged:

  $RMT_BASE_DIR/outputs/$RMT_PROJECT/<model>/train/models/ckpt.<N>.pt       tokenizer
  $RMT_BASE_DIR/outputs/$RMT_PROJECT/dit_<model>/train/models/ckpt.<N>.pt   DiT (+ its tokenizer)

  python tools/download_checkpoints.py --list
  python tools/download_checkpoints.py                                   # all 30 configurations
  python tools/download_checkpoints.py --models discrete.rqvae.f8_d4_e16384
  python tools/download_checkpoints.py --kind tokenizer                  # tokenizers only

A DiT needs its tokenizer, so `--kind dit` also fetches the tokenizer file.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request

import torch
from safetensors.torch import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import paths  # noqa: E402

MANIFEST = os.path.join(ROOT, "tools", "checkpoints.json")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, dst, digest):
    if os.path.isfile(dst) and sha256(dst) == digest:
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    print(f"  downloading {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 22)
    if sha256(tmp) != digest:
        os.remove(tmp)
        raise RuntimeError(f"checksum mismatch for {os.path.basename(dst)}")
    os.replace(tmp, dst)
    return dst


def run_dir(name):
    return os.path.join(paths.BASE_DIR, "outputs", paths.PROJECT, name, "train", "models")


def install(model, entry, kinds, base_url, cache):
    files = {k: fetch(f"{base_url}/{entry[k]['file']}", os.path.join(cache, entry[k]["file"]),
                      entry[k]["sha256"])
             for k in ("tokenizer", "dit") if k in kinds}
    tok = load_file(files["tokenizer"])
    out = run_dir(model)
    os.makedirs(out, exist_ok=True)
    torch.save({"model": tok}, os.path.join(out, f"ckpt.{entry['tokenizer']['ckpt']}.pt"))
    print(f"  {model}: tokenizer -> {out}")
    if "dit" in kinds:
        gen = load_file(files["dit"])
        gen.update({f"first_stage.{k}": v for k, v in tok.items()})
        out = run_dir(f"dit_{model}")
        os.makedirs(out, exist_ok=True)
        torch.save({"model": gen, "steps": entry["dit"]["steps"]},
                   os.path.join(out, f"ckpt.{entry['dit']['ckpt']}.pt"))
        print(f"  {model}: DiT       -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=None, help="configurations (default: all 30)")
    ap.add_argument("--kind", choices=["tokenizer", "dit", "all"], default="all")
    ap.add_argument("--cache", default=os.path.join(paths.BASE_DIR, "release_weights"),
                    help="where the downloaded .safetensors files are kept")
    ap.add_argument("--list", action="store_true", help="list the available configurations")
    args = ap.parse_args()

    manifest = json.load(open(MANIFEST))
    if args.list:
        for m, e in manifest["models"].items():
            print(f"{m:38s} tokenizer {e['tokenizer']['bytes'] / 1e6:5.0f} MB   "
                  f"dit {e['dit']['bytes'] / 1e6:5.0f} MB")
        return
    models = args.models or list(manifest["models"])
    unknown = [m for m in models if m not in manifest["models"]]
    if unknown:
        sys.exit(f"unknown configuration(s): {', '.join(unknown)} (see --list)")
    kinds = {"tokenizer"} if args.kind == "tokenizer" else {"tokenizer", "dit"}
    base_url = f"https://github.com/{manifest['repo']}/releases/download/{manifest['tag']}"
    for m in models:
        install(m, manifest["models"][m], kinds, base_url, args.cache)


if __name__ == "__main__":
    main()
