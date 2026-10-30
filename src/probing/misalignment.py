"""
Codebook–latent misalignment visualization (Fig. 3 in DiVeQ, Vali et al. 2026).

For each (discrete model, dataset) we:
  - Run the encoder on the test split and collect the pre-quant latent vectors
    P_z = {z_n} as the "gray points".
  - Pull the learned codebook C_z = {c_k} from the quantizer as the "red crosses".
    For residual quantizers we use the *first* level's codebook — that's the
    only one whose distribution should align with z; deeper levels see residuals.
  - Project P_z ∪ C_z jointly with t-SNE and plot.
  - Report distortion per bit: mean ||z - z_q||^2 divided by the number of bits
    a single spatial token consumes (sum of log2(n_e) across active codebooks).

Continuous and token-* families are skipped (no codebook to misalign with).
"""
import os
import json
import time
import math
import numpy as np
import torch
from tqdm import tqdm

from .loader import load_first_stage
from .datasets import build_probe_loaders, ALL_2D


# ---------------------------------------------------------------------------
# codebook + bits/token extraction, dispatched on quantizer class name
# ---------------------------------------------------------------------------

def _is_vq_family(family):
    return family == "discrete"


def _codebook_of(level):
    """
    Return one codebook tensor (n_e, e_dim) for a single non-residual quantizer.
    """
    cls = type(level).__name__
    if cls in ("VectorQuantizer", "VectorQuantizer2", "SimpleQINCo"):
        # nn.Embedding (VQ) or ImplicitEmbedding (SimpleQINCo) — both expose .weight
        return level.embedding.weight.detach()
    if cls in ("SimVQ", "DiVeQ", "SFDiVeQ"):
        # SimVQ.embedding is a property; DiVeQ/SFDiVeQ expose `embedding` as a
        # raw (n_e, e_dim) tensor. Both .detach() cleanly.
        return level.embedding.detach()
    if cls in ("LookupFreeQuantizer", "BinarySphericalQuantizer"):
        return level.codebook.detach()
    raise NotImplementedError(
        f"Don't know how to extract codebook from quantizer class '{cls}'."
    )


def _bits_of(level):
    """Bits consumed per token by a single non-residual quantizer."""
    cls = type(level).__name__
    if cls in ("VectorQuantizer", "VectorQuantizer2", "SimpleQINCo", "SimVQ",
               "DiVeQ", "SFDiVeQ"):
        return math.log2(int(level.n_e))
    if cls in ("LookupFreeQuantizer", "BinarySphericalQuantizer"):
        # token_size = log2(codebook_size); both attrs exist on LFQ
        return float(level.token_size)
    raise NotImplementedError(
        f"Don't know bits/token for quantizer class '{cls}'."
    )


def get_codebook_and_bits(quantizer):
    """
    Returns:
      codebook: np.ndarray (K, D) — first-level codebook for plotting
      bits_per_token: float — total bits/token across all residual levels
      meta: dict — quantizer type, num_levels, per-level codebook sizes
    """
    cls = type(quantizer).__name__
    if cls in ("ResidualQuantizer", "QincoResidualQuantizer"):
        levels = list(quantizer.levels)
        first_cb = _codebook_of(levels[0]).cpu().numpy()
        total_bits = sum(_bits_of(lv) for lv in levels)
        meta = {
            "quantizer_class": cls,
            "num_levels": len(levels),
            "level_classes": [type(lv).__name__ for lv in levels],
            "level_codebook_sizes": [int(getattr(lv, "n_e", -1)) for lv in levels],
            "level_num_codes": [_num_codes_of(lv) for lv in levels],
            "level_bits": [_bits_of(lv) for lv in levels],
            "plotted_level": 0,
        }
        return first_cb, total_bits, meta

    cb = _codebook_of(quantizer).cpu().numpy()
    bits = _bits_of(quantizer)
    meta = {
        "quantizer_class": cls,
        "num_levels": 1,
        "level_classes": [cls],
        "level_codebook_sizes": [int(cb.shape[0])],
        "level_num_codes": [_num_codes_of(quantizer)],
        "level_bits": [bits],
        "plotted_level": 0,
    }
    return cb, bits, meta


# ---------------------------------------------------------------------------
# pre-quant latent extraction (loop over test split, no pooling)
# ---------------------------------------------------------------------------

def _first_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
        return out[0]
    if hasattr(out, "mode"):
        return out.mode()
    if hasattr(out, "sample"):
        return out.sample()
    raise TypeError(f"Cannot extract latent tensor from {type(out).__name__}")


def _extract_plotted_level_indices(encode_out, quantizer_class):
    """
    Return a 1-D tensor of selected codebook indices for the level we plot
    (= first level of a residual stack, or the only level otherwise).

    medlat's `model.encode()` returns `(quant, loss, idx)` with the indices
    directly at position [2]:
      - single-codebook (VQ/SimVQ/SimpleQINCo/DiVeQ/SFDiVeQ): tensor
      - LFQ / BSQ:                                            spatial int tensor
      - ResidualQuantizer:                                    list[per-level tensors]
    Falls back to the legacy medlat layout where [2] was a
    (perplexity, encodings, indices) tuple.
    """
    if not isinstance(encode_out, (tuple, list)) or len(encode_out) < 3:
        return None
    third = encode_out[2]
    if quantizer_class in ("ResidualQuantizer", "QincoResidualQuantizer"):
        if isinstance(third, (list, tuple)) and len(third) and torch.is_tensor(third[0]):
            return third[0].reshape(-1)          # first (coarsest) level
        return None
    if torch.is_tensor(third):
        return third.reshape(-1)
    # legacy medlat: (perplexity, encodings, indices)
    if isinstance(third, (tuple, list)) and len(third) >= 3 and torch.is_tensor(third[2]):
        return third[2].reshape(-1)
    return None


def _extract_all_level_indices(encode_out, quantizer_class):
    """Return a list of 1-D index tensors, one per (residual) level.

    For single-codebook quantizers this is a one-element list. Used to build
    per-stage code-usage histograms for the empirical-entropy denominator
    H(C) = sum_l H(C_l), the counterpart to the allocated budget B = sum_l log2 K_l.
    """
    if not isinstance(encode_out, (tuple, list)) or len(encode_out) < 3:
        return None
    third = encode_out[2]
    if quantizer_class in ("ResidualQuantizer", "QincoResidualQuantizer"):
        if isinstance(third, (list, tuple)) and len(third) and torch.is_tensor(third[0]):
            return [t.reshape(-1) for t in third if torch.is_tensor(t)]
        return None
    if torch.is_tensor(third):
        return [third.reshape(-1)]
    if isinstance(third, (tuple, list)) and len(third) >= 3 and torch.is_tensor(third[2]):
        return [third[2].reshape(-1)]
    return None


def _num_codes_of(level):
    """Number of codebook entries K for a single quantizer level."""
    cls = type(level).__name__
    if cls in ("VectorQuantizer", "VectorQuantizer2", "SimpleQINCo", "SimVQ",
               "DiVeQ", "SFDiVeQ"):
        return int(level.n_e)
    if cls in ("LookupFreeQuantizer", "BinarySphericalQuantizer"):
        return int(round(2 ** float(level.token_size)))
    raise NotImplementedError(f"Don't know #codes for quantizer class '{cls}'.")


def _entropy_bits(counts):
    """Shannon entropy (in bits) of an empirical count vector; 0 if unused."""
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


@torch.no_grad()
def collect_pre_and_post(model, loader, device, max_latents,
                         quantizer_class=None, level_num_codes=None,
                         autocast_dtype=torch.bfloat16):
    """
    Stream the test loader, encode each batch to (pre, post) latents
    (B, C, H, W), flatten spatially to (B*H*W, C), and accumulate up to
    `max_latents` vectors per side (reservoir-style by random subsampling
    each batch). Also collects the set of codebook indices that were ever
    selected on the plotted level so the caller can subset C_z to live codes.

    Returns:
      pre: (N, D) numpy float32
      post: (N, D) numpy float32
      sse: float — sum of squared (z - z_q) over ALL positions seen (not just
           the kept subset); used for an unbiased distortion estimate.
      count: int — number of positions contributing to `sse`.
      used_indices: np.ndarray[int64] | None — sorted unique codebook indices
           selected on the plotted level across the entire split (or None if
           the quantizer's encode() return doesn't expose indices in a form
           we recognize).
    """
    rng = np.random.default_rng(0)
    pre_kept, post_kept = [], []
    sse_total = 0.0
    count_total = 0
    used_idx_chunks = []
    # per-level code-usage histograms for the empirical entropy H(C) = sum_l H(C_l)
    code_counts = ([np.zeros(int(K), dtype=np.int64) for K in level_num_codes]
                   if level_num_codes else None)

    # how many spatial positions to keep per batch — proportional to batch size
    # over the dataset, but capped so the running buffer doesn't exceed
    # max_latents until the loader is exhausted.
    n_batches = max(len(loader), 1)
    per_batch_keep = max(1, max_latents // n_batches)

    for batch in tqdm(loader, leave=False, dynamic_ncols=True):
        img = batch[0] if isinstance(batch, (list, tuple)) else batch["image"]
        img = img.to(device, non_blocking=True)

        if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                pre_out = model.encode_to_prequant(img)
                post_out = model.encode(img)
        else:
            pre_out = model.encode_to_prequant(img)
            post_out = model.encode(img)

        z_pre = _first_tensor(pre_out).float()
        z_post = _first_tensor(post_out).float()

        # Collect codebook indices selected on the plotted level (filter the
        # -1 sentinel ResidualQuantizer uses when a level was dropped).
        if quantizer_class is not None:
            idx = _extract_plotted_level_indices(post_out, quantizer_class)
            if idx is not None:
                idx = idx[idx >= 0]
                if idx.numel() > 0:
                    used_idx_chunks.append(torch.unique(idx).detach().cpu())

        # Accumulate per-level code-usage counts for the entropy denominator.
        if code_counts is not None:
            all_idx = _extract_all_level_indices(post_out, quantizer_class)
            if all_idx is not None:
                for lv, lv_idx in enumerate(all_idx):
                    if lv >= len(code_counts):
                        break
                    K = code_counts[lv].shape[0]
                    li = lv_idx[(lv_idx >= 0) & (lv_idx < K)].detach().cpu()
                    if li.numel() > 0:
                        code_counts[lv] += np.bincount(
                            li.numpy().astype(np.int64), minlength=K)[:K]

        # Handle (B, C, H, W) and (B, C, D, H, W); flatten all non-batch,
        # non-channel dims.
        if z_pre.dim() < 3:
            raise RuntimeError(f"Unexpected latent rank {z_pre.dim()} for pre-quant tensor.")
        b, c = z_pre.shape[0], z_pre.shape[1]
        zp = z_pre.reshape(b, c, -1).permute(0, 2, 1).reshape(-1, c).contiguous()
        zq = z_post.reshape(b, c, -1).permute(0, 2, 1).reshape(-1, c).contiguous()

        diff = zp - zq
        sse_total += float((diff * diff).sum().item())
        count_total += int(diff.numel() / c)

        n = zp.shape[0]
        k = min(per_batch_keep, n)
        if k < n:
            idx2 = rng.choice(n, size=k, replace=False)
            idx_t = torch.as_tensor(idx2, device=zp.device)
            zp = zp.index_select(0, idx_t)
            zq = zq.index_select(0, idx_t)

        pre_kept.append(zp.cpu().numpy().astype(np.float32))
        post_kept.append(zq.cpu().numpy().astype(np.float32))

    pre = np.concatenate(pre_kept, axis=0) if pre_kept else np.zeros((0, 0), dtype=np.float32)
    post = np.concatenate(post_kept, axis=0) if post_kept else np.zeros((0, 0), dtype=np.float32)

    if pre.shape[0] > max_latents:
        idx = rng.choice(pre.shape[0], size=max_latents, replace=False)
        pre = pre[idx]
        post = post[idx]

    used_indices = (
        torch.unique(torch.cat(used_idx_chunks)).numpy().astype(np.int64)
        if used_idx_chunks else None
    )
    return pre, post, sse_total, count_total, used_indices, code_counts


# ---------------------------------------------------------------------------
# latent / codebook geometry (stable rank + anisotropy)
# ---------------------------------------------------------------------------

def _latent_geometry(X, top_k=8):
    """
    Compute centered-SVD-derived geometry metrics for an (N, D) matrix.

    Reports:
      - stable_rank     = Σ σᵢ² / σ_max²        ∈ [1, D].  D = isotropic, 1 = rank-1.
      - effective_rank  = exp(H(p)) where p = σᵢ² / Σσⱼ²; same range as stable_rank,
                          based on the entropy of the singular spectrum
                          (Roy & Vetterli, 2007).
      - anisotropy      = σ_max² / σ_min²       high = squashed distribution.
      - log10_anisotropy: more readable.

    Centering is applied before the eigendecomp so we measure the *spread*
    of the distribution rather than its offset from the origin (a cloud
    centered far from zero with a tiny spread would otherwise read as
    rank-1 even if the spread itself is isotropic).

    Implementation: eigvalsh on the D×D covariance is O(D³) and independent
    of N — much cheaper than a full N×D SVD when N >> D (always the case
    here: f8_d4 has D=4, even token tokenizers stay ≤ 64).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.size == 0:
        return {"dim": 0, "n_samples": 0, "stable_rank": 0.0,
                "effective_rank": 0.0, "anisotropy": float("nan"),
                "log10_anisotropy": float("nan"), "sigma_sq_top": []}
    n, d = X.shape
    if n < 2:
        return {"dim": int(d), "n_samples": int(n), "stable_rank": 0.0,
                "effective_rank": 0.0, "anisotropy": float("nan"),
                "log10_anisotropy": float("nan"), "sigma_sq_top": []}
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / (n - 1)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    s2 = eigvals[::-1]    # descending = σ_max² down to σ_min²
    s2_sum = float(s2.sum())
    s2_max = float(s2[0])
    s2_min = float(s2[-1])
    if s2_max <= 0:
        return {"dim": int(d), "n_samples": int(n), "stable_rank": 0.0,
                "effective_rank": 0.0, "anisotropy": float("nan"),
                "log10_anisotropy": float("nan"),
                "sigma_sq_top": s2[:top_k].tolist()}
    stable_rank = s2_sum / s2_max
    p = s2 / s2_sum
    nz = p > 1e-30
    eff_rank = float(np.exp(-(p[nz] * np.log(p[nz])).sum()))
    if s2_min > 0:
        anisotropy = s2_max / s2_min
        log10_anis = float(np.log10(anisotropy))
    else:
        anisotropy = float("inf")
        log10_anis = float("inf")
    return {
        "dim": int(d),
        "n_samples": int(n),
        "stable_rank": float(stable_rank),
        "effective_rank": float(eff_rank),
        "anisotropy": float(anisotropy),
        "log10_anisotropy": log10_anis,
        "sigma_sq_top": s2[:top_k].tolist(),
    }


def _format_geo(label, geo):
    """Compact one-line summary of a _latent_geometry result, for log output."""
    sr = geo.get("stable_rank", 0.0)
    er = geo.get("effective_rank", 0.0)
    d = geo.get("dim", 0)
    anis = geo.get("anisotropy", float("nan"))
    return (f"{label}: dim={d} stable_rank={sr:.2f}/{d} "
            f"eff_rank={er:.2f} anisotropy={anis:.3g}")


# ---------------------------------------------------------------------------
# t-SNE + plotting
# ---------------------------------------------------------------------------

def _joint_tsne(latents, codebook, perplexity=30, seed=0):
    """
    Run t-SNE on stacked [latents; codebook] and split the result.
    Returns (latents_2d, codebook_2d).
    """
    from sklearn.manifold import TSNE
    n_l = latents.shape[0]
    n_c = codebook.shape[0]
    X = np.concatenate([latents, codebook], axis=0).astype(np.float32)

    # cap perplexity at (n-1)/3 as t-SNE requires
    eff_perp = float(min(perplexity, max(5, (X.shape[0] - 1) // 3)))
    tsne_kwargs = dict(
        n_components=2,
        perplexity=eff_perp,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    # sklearn renamed n_iter -> max_iter in 1.5; support both.
    try:
        import inspect
        sig = inspect.signature(TSNE.__init__).parameters
        if "max_iter" in sig:
            tsne_kwargs["max_iter"] = 1000
        elif "n_iter" in sig:
            tsne_kwargs["n_iter"] = 1000
    except Exception:
        pass
    tsne = TSNE(**tsne_kwargs)
    Y = tsne.fit_transform(X)
    return Y[:n_l], Y[n_l:n_l + n_c]


def _plot_misalignment(lat_2d, cb_2d, *, title, dist_per_bit, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    ax.scatter(lat_2d[:, 0], lat_2d[:, 1], s=6, c="steelblue", alpha=0.55,
               linewidths=0, label=r"$\mathcal{P}_z$ (latents)")
    ax.scatter(cb_2d[:, 0], cb_2d[:, 1], s=6, c="crimson", marker="x",
               linewidths=0.7, label=r"$\mathcal{C}_z$ (codebook, used)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    if dist_per_bit is not None and math.isfinite(dist_per_bit):
        if dist_per_bit >= 1e-2:
            txt = f"D/H(C) = {dist_per_bit:.3g}"
        else:
            txt = f"D/H(C) = {dist_per_bit:.2e}"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                ha="left", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.85))
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_pooled_misalignment(lat_2d, cb_2d, dataset_ids, dataset_names, *,
                              title, dist_per_bit, color_by_dataset, out_path):
    """
    Like `_plot_misalignment` but for a pooled (all-datasets) cell. When
    color_by_dataset is True, each dataset's latent points get a distinct
    color from `tab20`, with a compact legend below the plot. Codebook
    entries stay as red ×.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    if color_by_dataset:
        cmap = plt.get_cmap("tab20")
        for ds_idx, ds_name in enumerate(dataset_names):
            mask = dataset_ids == ds_idx
            if not mask.any():
                continue
            ax.scatter(lat_2d[mask, 0], lat_2d[mask, 1], s=6,
                       c=[cmap(ds_idx % 20)], alpha=0.55, linewidths=0,
                       label=ds_name)
    else:
        ax.scatter(lat_2d[:, 0], lat_2d[:, 1], s=6, c="steelblue", alpha=0.55,
                   linewidths=0, label=r"$\mathcal{P}_z$ (latents)")
    ax.scatter(cb_2d[:, 0], cb_2d[:, 1], s=6, c="crimson", marker="x",
               linewidths=0.7, label=r"$\mathcal{C}_z$ (used codes)")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    if dist_per_bit is not None and math.isfinite(dist_per_bit):
        if dist_per_bit >= 1e-2:
            txt = f"D/H(C) = {dist_per_bit:.3g}"
        else:
            txt = f"D/H(C) = {dist_per_bit:.2e}"
        ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                ha="left", va="top", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.85))
    # 13 entries (12 datasets + codebook) is too many for an inset legend;
    # park it below the axes in a 4-column grid.
    if color_by_dataset and len(dataset_names) > 6:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.02),
                  ncol=4, fontsize=7, frameon=True)
    else:
        ax.legend(loc="lower right", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def _aggregate_grid(per_model_results, out_path):
    """
    Build a single (n_models x n_datasets) grid figure stitching the per-cell
    PNGs together so multiple models are easy to eyeball side by side.
    Reads files lazily via PIL so we don't keep all rasters in memory.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    rows = [r for r in per_model_results if r["cells"]]
    if not rows:
        return
    ds_order = []
    seen = set()
    for r in rows:
        for c in r["cells"]:
            if c["dataset"] not in seen:
                ds_order.append(c["dataset"]); seen.add(c["dataset"])

    n_rows, n_cols = len(rows), len(ds_order)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.2 * n_cols, 2.4 * n_rows),
                             squeeze=False, dpi=120)
    for i, r in enumerate(rows):
        cells_by_ds = {c["dataset"]: c for c in r["cells"]}
        for j, ds in enumerate(ds_order):
            ax = axes[i][j]
            ax.set_xticks([]); ax.set_yticks([])
            if ds in cells_by_ds and os.path.exists(cells_by_ds[ds]["plot_path"]):
                img = Image.open(cells_by_ds[ds]["plot_path"])
                ax.imshow(img)
            if i == 0:
                ax.set_title(ds, fontsize=8)
            if j == 0:
                ax.set_ylabel(r["model"], fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_misalignment(cfg):
    """
    cfg keys (DictConfig-style):
      models: list of medlat model names (must be discrete.*)
      datasets: list of MedMNIST 2D dataset names, or None for all 12
      base_dir, project, data_root, image_size
      batch_size, num_workers, device
      ckpt: int or None
      split: 'test' (default) | 'val' | 'train'
      max_latents: int — cap on plotted latent points per cell (default 5000)
      max_codebook: int — cap on plotted codebook entries per cell (default 4096)
      perplexity: t-SNE perplexity (default 30)
      seed: int (default 0)
      run_name: optional override; default 'misalign_YYYYMMDD_HHMMSS'
      aggregate_dir: optional override; default <base_dir>/probing/<run_name>/
    """
    device = cfg.get("device", "cuda")
    image_size = int(cfg.get("image_size", 224))
    bs = int(cfg.get("batch_size", 256))
    num_workers = int(cfg.get("num_workers", 8))
    base_dir = cfg["base_dir"]
    project = cfg.get("project", "tokenizers")
    data_root = cfg["data_root"]
    split = cfg.get("split", "test")
    max_latents = int(cfg.get("max_latents", 5000))
    max_codebook = int(cfg.get("max_codebook", 4096))
    perplexity = float(cfg.get("perplexity", 30))
    seed = int(cfg.get("seed", 0))
    pool_datasets = bool(cfg.get("pool_datasets", True))
    color_by_dataset = bool(cfg.get("color_by_dataset", True))
    run_name = cfg.get("run_name") or time.strftime("misalign_%Y%m%d_%H%M%S")
    # Per-model outputs live alongside the other evaluation artifacts, in
    # <model_dir>/evaluation/<out_subdir>/. out_subdir is configurable but
    # defaults to "codebook_alignment".
    out_subdir = cfg.get("out_subdir") or "codebook_alignment"

    models = list(cfg["models"])
    datasets = list(cfg.get("datasets") or ALL_2D)

    # Cross-model grid only makes sense when several models run in one
    # invocation. In the SLURM sweep each task does a single model, so the
    # grid is skipped unless aggregate_dir is explicitly set.
    aggregate_dir = cfg.get("aggregate_dir")

    print(f"[misalign] run_name={run_name}")
    print(f"[misalign] models ({len(models)}): {models}")
    print(f"[misalign] datasets ({len(datasets)}): {datasets}")
    print(f"[misalign] per-model out: <model_dir>/evaluation/{out_subdir}/")

    rng = np.random.default_rng(seed)
    all_results = []
    aggregate_full = {"run_name": run_name, "config": dict(cfg), "results": []}

    for model_name in models:
        print(f"\n========== MODEL: {model_name} ==========")
        info = load_first_stage(
            model_name, base_dir=base_dir, project=project, ckpt=cfg.get("ckpt"),
            img_size=image_size, device=device,
        )
        model = info["model"]
        family = info["family"]
        if not _is_vq_family(family):
            print(f"[misalign] skipping {model_name}: family '{family}' has no discrete codebook.")
            continue
        if not hasattr(model, "quantizer") or not hasattr(model, "encode_to_prequant"):
            print(f"[misalign] skipping {model_name}: missing quantizer/encode_to_prequant.")
            continue

        try:
            cb, bits_per_token, q_meta = get_codebook_and_bits(model.quantizer)
        except NotImplementedError as e:
            print(f"[misalign] skipping {model_name}: {e}")
            continue
        print(f"[misalign] quantizer={q_meta['quantizer_class']} "
              f"levels={q_meta['num_levels']} bits/token={bits_per_token:.2f} "
              f"first-level codebook={cb.shape}")

        per_model_dir = os.path.join(info["model_dir"], "evaluation", out_subdir)
        os.makedirs(per_model_dir, exist_ok=True)
        n_codes_total = int(cb.shape[0])

        # Resume from an earlier run: keep existing cells from results.json so a
        # second invocation only fills in the missing datasets.
        existing_results_path = os.path.join(per_model_dir, "results.json")
        if os.path.exists(existing_results_path):
            try:
                with open(existing_results_path) as f:
                    prev = json.load(f)
                model_result = prev
                model_result["model"] = model_name
                model_result["family"] = family
                model_result["ckpt_step"] = info["ckpt_step"]
                model_result["quantizer"] = q_meta
                model_result["bits_per_token"] = bits_per_token
                model_result.setdefault("cells", [])
                print(f"[misalign] resuming: {len(model_result['cells'])} cells already on disk.")
            except Exception as e:
                print(f"[misalign] WARN couldn't parse prior results.json ({e}); starting fresh.")
                model_result = {
                    "model": model_name, "family": family, "ckpt_step": info["ckpt_step"],
                    "quantizer": q_meta, "bits_per_token": bits_per_token, "cells": [],
                }
        else:
            model_result = {
                "model": model_name, "family": family, "ckpt_step": info["ckpt_step"],
                "quantizer": q_meta, "bits_per_token": bits_per_token, "cells": [],
            }
        if pool_datasets:
            # ---- POOLED MODE: one t-SNE per model over the union of datasets ----
            pooled_png = os.path.join(per_model_dir, "pooled.png")
            done_pooled = any(
                c.get("dataset") == "pooled" and os.path.exists(c.get("plot_path", ""))
                for c in model_result["cells"]
            )
            if done_pooled:
                print(f"[misalign] skip {model_name} (pooled.png already exists).")
            else:
                # Split the latent budget evenly across datasets so a big test
                # set doesn't dominate; small datasets fill less of their quota.
                per_ds_budget = max(1, max_latents // max(len(datasets), 1))

                pooled_pre = []
                pooled_ds_ids = []
                used_set = set()
                sse_total = 0.0
                count_total = 0
                pooled_counts = None
                t_pool = time.time()
                for ds_idx, ds_name in enumerate(datasets):
                    print(f"\n--- {model_name} :: pool({ds_name}, budget={per_ds_budget}) ---")
                    loaders, _spec = build_probe_loaders(
                        ds_name, root=data_root, batch_size=bs, num_workers=num_workers,
                        image_size=image_size, download=False,
                    )
                    pre_ds, _post, sse_ds, count_ds, used_ds, counts_ds = collect_pre_and_post(
                        model, loaders[split], device=device, max_latents=per_ds_budget,
                        quantizer_class=q_meta["quantizer_class"],
                        level_num_codes=q_meta.get("level_num_codes"),
                    )
                    if count_ds == 0:
                        print(f"[misalign] WARN no positions for {ds_name}; skipping.")
                        continue
                    pooled_pre.append(pre_ds)
                    pooled_ds_ids.append(np.full(pre_ds.shape[0], ds_idx, dtype=np.int32))
                    if used_ds is not None:
                        used_ds = used_ds[(used_ds >= 0) & (used_ds < n_codes_total)]
                        used_set.update(used_ds.tolist())
                    if counts_ds is not None:
                        if pooled_counts is None:
                            pooled_counts = [c.copy() for c in counts_ds]
                        else:
                            for lv in range(len(pooled_counts)):
                                pooled_counts[lv] += counts_ds[lv]
                    sse_total += sse_ds
                    count_total += count_ds

                if count_total == 0:
                    print(f"[misalign] WARN no positions collected for {model_name}; skipping pool.")
                else:
                    pre_all = np.concatenate(pooled_pre, axis=0)
                    ds_ids_all = np.concatenate(pooled_ds_ids, axis=0)
                    distortion = sse_total / max(count_total, 1)
                    dist_per_bit = distortion / max(bits_per_token, 1e-9)
                    # Empirical entropy denominator: H(C) = sum_l H(C_l) (bits used).
                    per_level_H = ([_entropy_bits(c) for c in pooled_counts]
                                   if pooled_counts is not None else [])
                    entropy_bits = float(sum(per_level_H))
                    dist_per_bit_entropy = distortion / max(entropy_bits, 1e-9)

                    used_arr = np.array(sorted(used_set), dtype=np.int64)
                    n_used = int(used_arr.size)
                    utilization = n_used / max(n_codes_total, 1)
                    cb_used = cb[used_arr] if n_used > 0 else cb
                    if cb_used.shape[0] > max_codebook:
                        idx_cb = rng.choice(cb_used.shape[0], size=max_codebook, replace=False)
                        cb_plot = cb_used[idx_cb]
                    else:
                        cb_plot = cb_used
                    extract_time = time.time() - t_pool

                    if cb_plot.shape[0] == 0:
                        print(f"[misalign] WARN no codes used for {model_name}; skipping plot.")
                    else:
                        geo_lat = _latent_geometry(pre_all)
                        geo_cb = _latent_geometry(cb_used)
                        print(f"[misalign] pooled {pre_all.shape[0]} latents across "
                              f"{len(datasets)} datasets ({extract_time:.1f}s); "
                              f"distortion={distortion:.4g}; D/bit(alloc)={dist_per_bit:.4g}; "
                              f"H(C)={entropy_bits:.3f} bits; D/H(C)={dist_per_bit_entropy:.4g}; "
                              f"codebook used {n_used}/{n_codes_total} "
                              f"({100*utilization:.1f}%) -> plotting {cb_plot.shape[0]}")
                        print(f"[misalign] {_format_geo('latents (P_z)', geo_lat)}")
                        print(f"[misalign] {_format_geo('used codes (C_z)', geo_cb)}")

                        t_tsne = time.time()
                        lat_2d, cb_2d = _joint_tsne(pre_all, cb_plot,
                                                    perplexity=perplexity, seed=seed)
                        tsne_time = time.time() - t_tsne

                        title = (f"{model_name}\npooled {split} latents "
                                 f"({len(datasets)} datasets)")
                        _plot_pooled_misalignment(
                            lat_2d, cb_2d, ds_ids_all, list(datasets),
                            title=title, dist_per_bit=dist_per_bit_entropy,
                            color_by_dataset=color_by_dataset, out_path=pooled_png,
                        )
                        print(f"[misalign] tsne+plot {tsne_time:.1f}s -> {pooled_png}")

                        # Replace any stale per-dataset cells from a previous
                        # non-pooled run; in pooled mode each model has exactly
                        # one "pooled" cell.
                        model_result["cells"] = [{
                            "dataset": "pooled",
                            "datasets": list(datasets),
                            "split": split,
                            "n_latents_plotted": int(pre_all.shape[0]),
                            "n_codebook_plotted": int(cb_plot.shape[0]),
                            "n_codes_used": int(n_used),
                            "n_codes_total": int(n_codes_total),
                            "codebook_utilization": float(utilization),
                            "n_positions_seen": int(count_total),
                            "distortion": float(distortion),
                            "distortion_per_bit": float(dist_per_bit),
                            "entropy_bits": float(entropy_bits),
                            "per_level_entropy_bits": [float(h) for h in per_level_H],
                            "distortion_per_bit_entropy": float(dist_per_bit_entropy),
                            "latent_geometry": geo_lat,
                            "codebook_geometry": geo_cb,
                            "extract_sec": float(extract_time),
                            "tsne_sec": float(tsne_time),
                            "plot_path": pooled_png,
                        }]
                        with open(os.path.join(per_model_dir, "results.json"), "w") as f:
                            json.dump(model_result, f, indent=2)
        else:
            # ---- PER-DATASET MODE: one t-SNE per (model, dataset) ----
            done_datasets = {c["dataset"] for c in model_result["cells"]
                             if os.path.exists(c.get("plot_path", ""))}

            for ds_name in datasets:
                if ds_name in done_datasets:
                    print(f"[misalign] skip {model_name} :: {ds_name} (PNG already exists).")
                    continue
                print(f"\n--- {model_name} :: {ds_name} ({split}) ---")
                loaders, _spec = build_probe_loaders(
                    ds_name, root=data_root, batch_size=bs, num_workers=num_workers,
                    image_size=image_size, download=False,
                )
                t0 = time.time()
                pre, post, sse, count, used, counts_ds = collect_pre_and_post(
                    model, loaders[split], device=device, max_latents=max_latents,
                    quantizer_class=q_meta["quantizer_class"],
                    level_num_codes=q_meta.get("level_num_codes"),
                )
                extract_time = time.time() - t0

                if count == 0:
                    print(f"[misalign] WARN no positions collected for {ds_name}; skipping cell.")
                    continue
                distortion = sse / max(count, 1)
                dist_per_bit = distortion / max(bits_per_token, 1e-9)
                per_level_H = ([_entropy_bits(c) for c in counts_ds]
                               if counts_ds is not None else [])
                entropy_bits = float(sum(per_level_H))
                dist_per_bit_entropy = distortion / max(entropy_bits, 1e-9)

                # Restrict the codebook to entries that were actually selected on
                # this split; that's the subset that matters for misalignment.
                if used is not None:
                    used = used[(used >= 0) & (used < n_codes_total)]
                    cb_used = cb[used]
                    n_used = int(used.size)
                else:
                    cb_used = cb
                    n_used = n_codes_total
                utilization = n_used / max(n_codes_total, 1)
                if cb_used.shape[0] > max_codebook:
                    idx_cb = rng.choice(cb_used.shape[0], size=max_codebook, replace=False)
                    cb_plot = cb_used[idx_cb]
                else:
                    cb_plot = cb_used
                if cb_plot.shape[0] == 0:
                    print(f"[misalign] WARN no codes used on {ds_name}; skipping cell.")
                    continue

                geo_lat = _latent_geometry(pre)
                geo_cb = _latent_geometry(cb_used)
                print(f"[misalign] kept {pre.shape[0]} latents ({extract_time:.1f}s); "
                      f"distortion={distortion:.4g}; D/bit={dist_per_bit:.4g}; "
                      f"codebook used {n_used}/{n_codes_total} ({100*utilization:.1f}%) "
                      f"-> plotting {cb_plot.shape[0]}")
                print(f"[misalign] {_format_geo('latents (P_z)', geo_lat)}")
                print(f"[misalign] {_format_geo('used codes (C_z)', geo_cb)}")

                t1 = time.time()
                lat_2d, cb_2d = _joint_tsne(pre, cb_plot, perplexity=perplexity, seed=seed)
                tsne_time = time.time() - t1

                plot_path = os.path.join(per_model_dir, f"{ds_name}.png")
                title = f"{model_name}\n{ds_name} ({split})"
                _plot_misalignment(lat_2d, cb_2d, title=title,
                                   dist_per_bit=dist_per_bit_entropy, out_path=plot_path)
                print(f"[misalign] tsne+plot {tsne_time:.1f}s -> {plot_path}")

                model_result["cells"].append({
                    "dataset": ds_name,
                    "split": split,
                    "n_latents_plotted": int(pre.shape[0]),
                    "n_codebook_plotted": int(cb_plot.shape[0]),
                    "n_codes_used": int(n_used),
                    "n_codes_total": int(n_codes_total),
                    "codebook_utilization": float(utilization),
                    "n_positions_seen": int(count),
                    "distortion": float(distortion),
                    "distortion_per_bit": float(dist_per_bit),
                    "entropy_bits": float(entropy_bits),
                    "per_level_entropy_bits": [float(h) for h in per_level_H],
                    "distortion_per_bit_entropy": float(dist_per_bit_entropy),
                    "latent_geometry": geo_lat,
                    "codebook_geometry": geo_cb,
                    "extract_sec": float(extract_time),
                    "tsne_sec": float(tsne_time),
                    "plot_path": plot_path,
                })

                with open(os.path.join(per_model_dir, "results.json"), "w") as f:
                    json.dump(model_result, f, indent=2)

        all_results.append(model_result)
        aggregate_full["results"] = all_results

        del model, info
        if device == "cuda":
            torch.cuda.empty_cache()

    # Cross-model grid + combined JSON only when an aggregate_dir is given
    # (i.e. several models run together). The sweep runs one model per task
    # and relies on the per-model <model_dir>/evaluation/<out_subdir>/ output.
    if aggregate_dir:
        os.makedirs(aggregate_dir, exist_ok=True)
        with open(os.path.join(aggregate_dir, "results.json"), "w") as f:
            json.dump(aggregate_full, f, indent=2)
        grid_path = os.path.join(aggregate_dir, "grid.png")
        try:
            _aggregate_grid(all_results, grid_path)
            print(f"\n[misalign] DONE. Grid: {grid_path}")
        except Exception as e:
            print(f"[misalign] grid plot failed (continuing): {e}")
    else:
        print(f"\n[misalign] DONE. Per-model outputs under "
              f"<model_dir>/evaluation/{out_subdir}/")
    return aggregate_full
