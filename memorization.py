#!/usr/bin/env python3
"""
memorization.py — generator memorization via calibrated nearest-neighbor copy
detection. For each (model, dataset) we embed the DiT samples, the TRAIN split
(the NN reference), and the TEST split (the calibration baseline) with one or
more feature extractors, then measure how close generated samples sit to the
training set relative to how close unseen real (test) images do.

Metrics per (model, dataset, extractor):
  copy_rate  = fraction of generated samples whose top-1 train-NN cosine
               similarity exceeds the 95th percentile of the TEST->train top-1
               similarities (memorization above the natural self-similarity floor).
  mem_ratio  = median Somepalli ratio  d1 / mean(d2..d_{k+1})  over generated
               samples (distance d = 1 - cos; lower => tighter copy of one
               specific training image).
  base_p95   = the test->train 95th-pct similarity threshold (the calibration).

Extractors:
  medmnist : per-dataset MedMNIST ResNet-18 (same as FMD), 512-d.
             Saturates on medical data (base_p95 ~ 0.994); use for comparison only.
  sscd     : SSCD copy-detection descriptor (torchscript), domain-agnostic, 512-d.
             Primary: calibrates well (base_p95 ~ 0.77).
  dinov2   : DINOv2 ViT-B/14 class token (torch.hub), 768-d.
             Richer semantic space; ablation to validate SSCD findings on medical data.
  both     : sscd + dinov2 (the main comparison).
  all      : medmnist + sscd + dinov2.
"""
import argparse, csv, glob, os
import numpy as np
import torch

from PIL import Image

from src import paths

ALL_2D = ["pathmnist","chestmnist","dermamnist","octmnist","pneumoniamnist","retinamnist",
          "breastmnist","bloodmnist","tissuemnist","organamnist","organcmnist","organsmnist"]
IMNET_MEAN = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
IMNET_STD  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)


def to_rgb_uint8(a):
    """(N,H,W) or (N,H,W,3) uint8 -> (N,H,W,3) uint8."""
    if a.ndim == 3:
        a = np.repeat(a[..., None], 3, axis=-1)
    return np.ascontiguousarray(a)


# ---------------- image sources (return uint8 (B,H,W,3)) ----------------
class NpzSource:
    def __init__(self, path, key):
        self.arr = np.load(path, mmap_mode="r")[key]
    def __len__(self): return len(self.arr)
    def batch(self, idx): return to_rgb_uint8(np.asarray(self.arr[idx]))

class PngSource:
    def __init__(self, d, cap=None):
        self.files = sorted(glob.glob(os.path.join(d, "*.png")))
        if cap: self.files = self.files[:cap]
    def __len__(self): return len(self.files)
    def batch(self, idx):
        return np.stack([np.asarray(Image.open(self.files[i]).convert("RGB")) for i in idx])


# ---------------- extractors ----------------
def load_medmnist_fe(dataset, wroot, arch, seed, device):
    from torch_fidelity.feature_extractor_medmnist import FeatureExtractorMedMNISTResNet
    wp = os.path.join(wroot, dataset, f"{arch}_224_{seed}.pth")
    fe = FeatureExtractorMedMNISTResNet(f"medmnist-{arch}", ["feature_vec"],
                                        feature_extractor_weights_path=wp).to(device).eval()
    def fn(u8):  # u8: (B,3,224,224) uint8 on device
        return fe(u8)[0]
    return fn

def load_sscd(sscd_path, device):
    model = torch.jit.load(sscd_path, map_location=device).eval()
    mean, std = IMNET_MEAN.to(device), IMNET_STD.to(device)
    def fn(u8):  # u8: (B,3,224,224) uint8 on device
        x = u8.float() / 255.0
        x = torch.nn.functional.interpolate(x, size=(288,288), mode="bilinear", align_corners=False)
        x = (x - mean) / std
        return model(x)
    return fn

def load_dinov2(device, hub_dir=None):
    """DINOv2 ViT-B/14 class token (768-d). Uses torch.hub cache at hub_dir."""
    if hub_dir:
        torch.hub.set_dir(hub_dir)
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14",
                           pretrained=True, verbose=False).to(device).eval()
    mean, std = IMNET_MEAN.to(device), IMNET_STD.to(device)
    @torch.no_grad()
    def fn(u8):  # u8: (B,3,224,224) uint8 on device
        x = u8.float() / 255.0
        x = (x - mean) / std
        return model(x)   # returns (B, 768) class token
    return fn

@torch.no_grad()
def embed(source, fn, device, bs=128):
    feats = []
    for s in range(0, len(source), bs):
        idx = range(s, min(s+bs, len(source)))
        u8 = torch.from_numpy(source.batch(idx)).permute(0,3,1,2).contiguous().to(device)
        f = fn(u8).float()
        f = torch.nn.functional.normalize(f, dim=1)   # cosine via dot
        feats.append(f.cpu())
    return torch.cat(feats)


# ---------------- NN copy-detection metrics ----------------
@torch.no_grad()
def topk_sims(query, ref, k, device, qchunk=2048):
    """For each query row, return top-(k) cosine sims to ref (descending)."""
    ref = ref.to(device)
    out = []
    for s in range(0, len(query), qchunk):
        q = query[s:s+qchunk].to(device)
        sims = q @ ref.t()                       # (b, Nref), cosine (normalized)
        tk = sims.topk(k, dim=1).values          # (b, k)
        out.append(tk.cpu())
    return torch.cat(out)

@torch.no_grad()
def top1_sims_and_indices(query, ref, device, qchunk=2048):
    """Return (sims, indices) arrays of shape (N,) — top-1 train NN per query."""
    ref = ref.to(device)
    sims_out, idx_out = [], []
    for s in range(0, len(query), qchunk):
        q = query[s:s+qchunk].to(device)
        res = (q @ ref.t()).topk(1, dim=1)
        sims_out.append(res.values.cpu())
        idx_out.append(res.indices.cpu())
    return (torch.cat(sims_out)[:,0].numpy(),
            torch.cat(idx_out)[:,0].numpy())


def save_flagged_pairs(gen_f, train_f, base_p95, gsrc, train_src,
                       out_dir, top_n, device):
    """Save the top_n most-similar flagged (generated, train-NN) pairs.

    Per pair, writes:
      top_NNN_gen.png        — the generated image
      top_NNN_train_nn.png   — its nearest training-set neighbour
      top_NNN_pair.png       — side-by-side composite (gen | train-NN)
    Plus pairs.csv with rank / indices / similarity.
    Returns total number of flagged images (before top_n cap).
    """
    all_sims, all_idx = top1_sims_and_indices(gen_f, train_f, device)

    flagged = np.where(all_sims > base_p95)[0]
    n_total = len(flagged)
    if n_total == 0:
        return 0

    # sort flagged by similarity descending
    flagged = flagged[np.argsort(-all_sims[flagged])][:top_n]

    os.makedirs(out_dir, exist_ok=True)
    meta = []
    for rank, gen_i in enumerate(flagged, 1):
        tr_i  = int(all_idx[gen_i])
        sim   = float(all_sims[gen_i])
        gen_u8 = gsrc.batch([int(gen_i)])[0]       # (H,W,3) uint8
        tr_u8  = train_src.batch([tr_i])[0]        # (H,W,3) uint8
        gen_img = Image.fromarray(gen_u8)
        tr_img  = Image.fromarray(tr_u8)
        gen_img.save(os.path.join(out_dir, f"top_{rank:03d}_gen.png"))
        tr_img.save(os.path.join(out_dir, f"top_{rank:03d}_train_nn.png"))
        # side-by-side: 2px grey separator
        W, H = gen_img.width, gen_img.height
        pair = Image.new("RGB", (W * 2 + 2, H), color=(180, 180, 180))
        pair.paste(gen_img, (0, 0))
        pair.paste(tr_img, (W + 2, 0))
        pair.save(os.path.join(out_dir, f"top_{rank:03d}_pair.png"))
        meta.append({"rank": rank, "gen_idx": int(gen_i), "train_idx": tr_i,
                     "similarity": round(sim, 5)})

    import csv as _csv
    with open(os.path.join(out_dir, "pairs.csv"), "w", newline="") as f:
        dw = _csv.DictWriter(f, fieldnames=["rank","gen_idx","train_idx","similarity"])
        dw.writeheader(); dw.writerows(meta)

    return n_total


def metrics(gen_f, train_f, test_f, k, device):
    kk = min(k, train_f.shape[0]-1)
    gen_tk  = topk_sims(gen_f,  train_f, kk+1, device)   # (Ng, k+1)
    test_tk = topk_sims(test_f, train_f, 1,   device)    # (Nt, 1)
    base_p95 = float(np.percentile(test_tk[:,0].numpy(), 95))
    s1 = gen_tk[:,0].numpy()
    copy_rate = float((s1 > base_p95).mean())
    # Somepalli ratio in distance space d=1-cos: d1 / mean(d2..d_{k+1})
    d = 1.0 - gen_tk.numpy()
    ratio = d[:,0] / (d[:,1:kk+1].mean(axis=1) + 1e-8)
    return dict(copy_rate=copy_rate, mem_ratio=float(np.median(ratio)),
                base_p95=base_p95, gen_top1_med=float(np.median(s1)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--extractor", choices=["medmnist","sscd","dinov2","both","all"],
                   default="both", help="both=sscd+dinov2, all=medmnist+sscd+dinov2")
    p.add_argument("--arch", default="resnet18"); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cap", type=int, default=10000, help="max generated samples / dataset")
    p.add_argument("--train_cap", type=int, default=0, help="0 = full train set as NN reference")
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--base_dir", default=paths.BASE_DIR,
                   help="Folder containing outputs/<project>/ ($RMT_BASE_DIR).")
    p.add_argument("--project", default=paths.PROJECT)
    p.add_argument("--weights_root", default=paths.MEDMNIST_WEIGHTS)
    p.add_argument("--medmnist_root", default=paths.MEDMNIST_ROOT)
    p.add_argument("--sscd_path", default=paths.SSCD_PATH)
    p.add_argument("--hub_dir", default=paths.TORCH_HUB_DIR,
                   help="torch.hub cache dir for DINOv2 ($TORCH_HUB_DIR; default: torch default).")
    p.add_argument("--save_flagged", default=None, metavar="DIR",
                   help="If set, save gen+train-NN image pairs for flagged samples here")
    p.add_argument("--top_n", type=int, default=20,
                   help="Max flagged pairs to save per (model, dataset, extractor)")
    p.add_argument("--out_csv", required=True); p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dev = args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu"
    datasets = args.datasets or ALL_2D
    base_proj = os.path.join(args.base_dir, "outputs", args.project)
    EXT_MAP = {"both": ["sscd","dinov2"], "all": ["medmnist","sscd","dinov2"]}
    exts = EXT_MAP.get(args.extractor, [args.extractor])

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    fields = ["model","dataset","extractor","copy_rate","mem_ratio","base_p95",
              "gen_top1_med","n_gen","n_train","n_test"]
    new = not os.path.exists(args.out_csv)
    fc = open(args.out_csv,"a",newline=""); w = csv.DictWriter(fc, fieldnames=fields)
    if new: w.writeheader()

    sscd_fn   = load_sscd(args.sscd_path, dev)       if "sscd"    in exts else None
    dinov2_fn = load_dinov2(dev, args.hub_dir)        if "dinov2"  in exts else None

    # cache per-dataset train/test embeddings (reused across models)
    cache = {}
    for ds in datasets:
        npz = os.path.join(args.medmnist_root, f"{ds}_224.npz")
        if not os.path.isfile(npz):
            print(f"[skip ds] {ds}: missing {npz}", flush=True); continue
        med_fn = load_medmnist_fe(ds, args.weights_root, args.arch, args.seed, dev) if "medmnist" in exts else None
        tr = NpzSource(npz, "train_images"); te = NpzSource(npz, "test_images")
        ntr = len(tr) if args.train_cap==0 else min(args.train_cap, len(tr))
        # build embedders dict
        embedders = {"medmnist": med_fn, "sscd": sscd_fn, "dinov2": dinov2_fn}
        cache[ds] = {"_src": tr}   # keep source for flagged-pair saving
        for e in exts:
            fn = embedders[e]
            class Sub:  # lightweight view over first ntr rows
                def __init__(s, src, n): s.src=src; s.n=n
                def __len__(s): return s.n
                def batch(s, idx): return s.src.batch(idx)
            tr_f = embed(Sub(tr, ntr), fn, dev); te_f = embed(te, fn, dev)
            cache[ds][e] = (tr_f, te_f, ntr, len(te))
            print(f"[emb] {ds}/{e}: train={ntr} test={len(te)}", flush=True)

        for m in args.models:
            sdir = os.path.join(base_proj, f"dit_{m}", "generation", ds, "sample")
            if not os.path.isdir(sdir) or not glob.glob(os.path.join(sdir,"*.png")):
                print(f"[skip] {m}/{ds}: no samples", flush=True); continue
            gsrc = PngSource(sdir, cap=args.cap)
            for e in exts:
                gen_f = embed(gsrc, embedders[e], dev)
                tr_f, te_f, ntr_, nte_ = cache[ds][e]
                r = metrics(gen_f, tr_f, te_f, args.k, dev)
                row = dict(model=m, dataset=ds, extractor=e, n_gen=len(gsrc),
                           n_train=ntr_, n_test=nte_, **{k:round(v,5) for k,v in r.items()})
                w.writerow(row); fc.flush()
                print(f"  {m} | {ds} | {e}: copy_rate={r['copy_rate']:.3f} "
                      f"mem_ratio={r['mem_ratio']:.3f} (base_p95={r['base_p95']:.3f}, "
                      f"gen_top1_med={r['gen_top1_med']:.3f})", flush=True)
                if args.save_flagged:
                    pair_dir = os.path.join(args.save_flagged, m, ds, e)
                    n_flag = save_flagged_pairs(gen_f, tr_f, r["base_p95"],
                                               gsrc, cache[ds]["_src"],
                                               pair_dir, args.top_n, dev)
                    print(f"    flagged: {n_flag} total, saved top-{args.top_n} -> {pair_dir}",
                          flush=True)
    fc.close(); print(f"\nWrote -> {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
