import argparse
import csv
import json
import math
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.io import write_png

import lpips as lpips_pkg
from torchmetrics.functional import structural_similarity_index_measure as ssim_fn

from src import paths
from src.probing.loader import load_first_stage
from src.probing.datasets import build_probe_loaders, ALL_2D


# =============================================================================
# Codebook helpers (kept inline so the math lives next to the call sites)
# =============================================================================

def resolve_quantizer(model):
    """
    Return the quantizer module of a medlat VQ model.

    IMPORTANT: medlat exposes the quantizer at `model.quantizer`. There is
    also a `model.quantize` *method* on VQModel — never probe that for
    attributes, every `hasattr` will silently succeed against the bound
    method and miss the real module.
    """
    q = getattr(model, "quantizer", None)
    if q is None:
        raise RuntimeError(
            f"evaluation: model (type={type(model).__name__}) has no `quantizer` "
            "attribute. Expected medlat VQModel layout."
        )
    return q


def level_offsets(quantizer):
    """
    For a ResidualQuantizer, return per-level start offsets so concatenated
    indices live in a disjoint global address space. None for non-residual,
    or [0]*K for shared-codebook RQs (no offset needed).
    """
    if not hasattr(quantizer, "levels"):
        return None
    levels = list(quantizer.levels)
    if not levels:
        return None
    if getattr(quantizer, "shared_codebook", False):
        return [0] * len(levels)
    sizes = [int(getattr(lv, "n_e", 0)) for lv in levels]
    if any(s <= 0 for s in sizes):
        return None
    offs = [0]
    for s in sizes[:-1]:
        offs.append(offs[-1] + s)
    return offs


def infer_codebook_size(quantizer):
    """
    Resolve the true codebook size K. Raises if it can't be determined —
    we'd rather fail loudly than silently fabricate a denominator (which is
    what the original VQUtilityEvaluation's `max_index + 1` fallback did).
    """
    if hasattr(quantizer, "levels") and hasattr(quantizer, "num_quantizers"):
        levels = list(quantizer.levels)
        if not levels:
            raise RuntimeError("evaluation: ResidualQuantizer has no levels.")
        if getattr(quantizer, "shared_codebook", False):
            n_e = int(getattr(levels[0], "n_e", 0))
            if n_e <= 0:
                raise RuntimeError("evaluation: shared-codebook RQ has n_e=0.")
            return n_e
        sizes = [int(getattr(lv, "n_e", 0)) for lv in levels]
        if any(s <= 0 for s in sizes):
            raise RuntimeError(f"evaluation: per-level n_e missing in RQ: {sizes}")
        return int(sum(sizes))
    for attr in ("n_e", "codebook_size"):
        val = getattr(quantizer, attr, None)
        if val is not None:
            return int(val)
    emb = getattr(quantizer, "embedding", None)
    if emb is not None:
        if hasattr(emb, "num_embeddings"):
            return int(emb.num_embeddings)
        if hasattr(emb, "weight") and hasattr(emb.weight, "shape"):
            return int(emb.weight.shape[0])
    raise RuntimeError(
        f"evaluation: can't infer codebook size from quantizer "
        f"{type(quantizer).__name__}. Extend infer_codebook_size."
    )


def flatten_indices(indices_raw, quantizer, K):
    """
    Flatten model.encode() info[2] into a 1-D long tensor on CPU. For
    ResidualQuantizer, offset per level so codes across levels are distinct.
    Strips the -1 dropped-level sentinel and bounds-checks against K.
    """
    if isinstance(indices_raw, (list, tuple)):
        offs = level_offsets(quantizer)
        if offs is None or len(offs) != len(indices_raw):
            raise RuntimeError(
                f"evaluation: got {len(indices_raw)} per-level index tensors but "
                f"quantizer={type(quantizer).__name__} doesn't expose matching levels."
            )
        shifted = [t.reshape(-1).long() + o for t, o in zip(indices_raw, offs)]
        flat = torch.cat(shifted, dim=0)
    elif torch.is_tensor(indices_raw):
        flat = indices_raw.reshape(-1).long()
    else:
        raise RuntimeError(
            f"evaluation: indices must be tensor or list of tensors "
            f"(got {type(indices_raw).__name__})."
        )
    flat = flat.cpu()
    flat = flat[flat >= 0]
    if flat.numel() > 0:
        m = int(flat.max().item())
        if m >= K:
            raise RuntimeError(
                f"evaluation: observed index {m} >= codebook_size {K}. "
                "Quantizer's declared size disagrees with encode() output."
            )
    return flat


def extract_indices_from_encode(encode_out, quantizer_class):
    """
    Pull the selected codebook indices from `model.encode()` == (quant, loss, idx).

    medlat returns the indices directly at position [2]:
      - single-codebook quantizers (VQ/SimVQ/SimpleQINCo/DiVeQ/SFDiVeQ/SoftVQ):
        an int tensor (flattened, or spatially-shaped for LFQ/BSQ)
      - residual quantizers (RQVAE/RSimpleQINCo): a list of per-level tensors
    Falls back to the legacy medlat layout where [2] was a
    (perplexity, encodings, indices) tuple.
    """
    if not isinstance(encode_out, (tuple, list)) or len(encode_out) < 3:
        return None
    third = encode_out[2]
    if quantizer_class in ("ResidualQuantizer", "QincoResidualQuantizer"):
        if isinstance(third, (list, tuple)) and len(third) and all(torch.is_tensor(t) for t in third):
            return list(third)
        return None
    if torch.is_tensor(third):
        return third
    # legacy medlat: (perplexity, encodings, indices)
    if isinstance(third, (tuple, list)) and len(third) >= 3 and torch.is_tensor(third[2]):
        return third[2]
    return None


def codebook_metrics_from_counts(counts, K):
    """
    Three cumulative codebook metrics from a pooled per-code count buffer.

    counts: 1-D float64 tensor of length K, holding bincount(selected indices)
            summed across every batch in the split.

    Returns:
      codebook_usage    : |unique codes selected| / K              ∈ [0, 1]
      codebook_entropy  : H(p) / log2(K)                          ∈ [0, 1]
      codebook_utility  : 1 / (1 + CV(p))                          ∈ (0, 1]
                          where p_k = counts_k / Σ counts (per-code frequency)
                          and CV = std(p) / mean(p). High = uniform usage.
    """
    total = float(counts.sum().item())
    K = int(K)
    if total <= 0 or K <= 0:
        return {"codebook_usage": 0.0, "codebook_entropy": 0.0,
                "codebook_utility": 0.0, "codebook_used": 0,
                "codebook_size": K, "codebook_total_assignments": 0}
    n_used = int((counts > 0).sum().item())
    usage = n_used / K
    probs = counts / total
    nz = probs > 0
    entropy = float(-(probs[nz] * torch.log2(probs[nz])).sum().item())
    max_entropy = math.log2(K) if K > 1 else 1.0
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    mean = float(probs.mean().item())
    std = float(probs.std().item())
    cv = std / (mean + 1e-10) if mean > 0 else 1.0
    utility = 1.0 / (1.0 + cv)
    return {
        "codebook_usage": usage,
        "codebook_entropy": norm_entropy,
        "codebook_utility": utility,
        "codebook_used": n_used,
        "codebook_size": K,
        "codebook_total_assignments": int(total),
    }


def extract_used_codebook_matrix(quantizer, used_mask):
    """
    Return the (n_used, D) matrix of codebook entries that were selected at
    least once. For ResidualQuantizer this returns rows from the first level
    only (consistent with the per-level offset scheme: used_mask[0:n_e] maps
    to level 0).
    """
    cls = type(quantizer).__name__
    if cls in ("VectorQuantizer", "VectorQuantizer2", "SimpleQINCo"):
        W = quantizer.embedding.weight.detach().cpu().numpy()
    elif cls in ("SimVQ", "DiVeQ", "SFDiVeQ"):
        W = quantizer.embedding.detach().cpu().numpy()
    elif cls in ("LookupFreeQuantizer", "BinarySphericalQuantizer"):
        W = quantizer.codebook.detach().cpu().numpy()
    elif cls in ("ResidualQuantizer", "QincoResidualQuantizer"):
        return extract_used_codebook_matrix(list(quantizer.levels)[0],
                                            used_mask[: int(list(quantizer.levels)[0].n_e)])
    else:
        raise RuntimeError(f"evaluation: no codebook extractor for {cls}.")
    used_mask = np.asarray(used_mask[: W.shape[0]], dtype=bool)
    return W[used_mask]


# =============================================================================
# Latent geometry (inline)
# =============================================================================

def latent_geometry(X):
    """
    One-shot centered-covariance metrics for an (N, D) matrix. Used when we
    already have the full matrix in memory — for streaming use
    `OnlineCovariance` instead.

      stable_rank    = Σ σ²ᵢ / σ²_max                    ∈ [1, D]
      effective_rank = exp(H(p)) where p_i = σ²ᵢ / Σσ²ⱼ ∈ [1, D]
      anisotropy     = σ²_max / σ²_min                   high = squashed
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.size == 0:
        return _empty_geo()
    n, d = X.shape
    if n < 2:
        return _empty_geo(d, n)
    Xc = X - X.mean(axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / (n - 1)
    return _metrics_from_cov(cov, n, d)


def _empty_geo(d=0, n=0):
    return {"dim": int(d), "n_samples": int(n), "stable_rank": 0.0,
            "effective_rank": 0.0, "anisotropy": float("nan"),
            "log10_anisotropy": float("nan")}


def _metrics_from_cov(cov, n, d):
    """
    Same metrics as `latent_geometry(X)` but starting from a (D, D) covariance
    matrix and a sample count. Lets `OnlineCovariance` reuse the math.
    """
    if n < 2 or d <= 0:
        return _empty_geo(d, n)
    eigvals = np.clip(np.linalg.eigvalsh(cov), 0.0, None)[::-1]
    s2_sum = float(eigvals.sum())
    s2_max = float(eigvals[0])
    s2_min = float(eigvals[-1])
    if s2_max <= 0:
        return _empty_geo(d, n)
    stable_rank = s2_sum / s2_max
    p = eigvals / s2_sum
    nz = p > 1e-30
    eff_rank = float(np.exp(-(p[nz] * np.log(p[nz])).sum()))
    if s2_min > 0:
        anisotropy = s2_max / s2_min
        log10_anis = float(np.log10(anisotropy))
    else:
        anisotropy = float("inf")
        log10_anis = float("inf")
    return {
        "dim": int(d), "n_samples": int(n),
        "stable_rank": float(stable_rank),
        "effective_rank": float(eff_rank),
        "anisotropy": float(anisotropy),
        "log10_anisotropy": log10_anis,
    }


class OnlineCovariance:
    """
    Streaming centered-covariance accumulator using Chan's parallel algorithm.

    Equivalent to running `latent_geometry` on every position in the stream,
    but with O(D²) memory regardless of N. Used to compute stable rank /
    anisotropy over the full test split without holding all latents in RAM.

    Accumulators are float64 so precision is fine up to ≳ 10⁸ samples.
    State lives on whichever device the first batch was on (typically CUDA),
    so we never bounce latents through CPU mid-loop; the final D×D covariance
    is moved to CPU for the eigvalsh call.
    """
    def __init__(self):
        self.n = 0
        self.dim = None
        self.device = None
        self.mean = None    # (D,) float64
        self.M2 = None      # (D, D) float64, Σ (x - μ_n)(x - μ_i)^T

    def update_batch(self, X):
        if not torch.is_tensor(X):
            X = torch.as_tensor(X)
        if X.ndim != 2 or X.shape[0] == 0:
            return
        if self.mean is None:
            self.dim = int(X.shape[1])
            self.device = X.device
            self.mean = torch.zeros(self.dim, dtype=torch.float64, device=self.device)
            self.M2 = torch.zeros(self.dim, self.dim, dtype=torch.float64,
                                  device=self.device)
        elif X.shape[1] != self.dim:
            raise RuntimeError(
                f"OnlineCovariance: dim mismatch ({X.shape[1]} vs {self.dim}).")
        X = X.to(self.device, dtype=torch.float64)
        n_a = self.n
        n_b = int(X.shape[0])
        mean_b = X.mean(dim=0)
        delta = mean_b - self.mean
        Xc = X - mean_b
        M2_b = Xc.T @ Xc
        new_n = n_a + n_b
        self.mean = self.mean + delta * (n_b / new_n)
        # Chan's correction term Σ_a Σ_b /(n_a+n_b) * δδ^T
        if n_a > 0:
            self.M2 = self.M2 + M2_b + torch.outer(delta, delta) * (n_a * n_b / new_n)
        else:
            self.M2 = self.M2 + M2_b
        self.n = new_n

    def metrics(self):
        if self.n < 2 or self.M2 is None:
            return _empty_geo(self.dim or 0, self.n)
        cov = (self.M2 / (self.n - 1)).cpu().numpy()
        return _metrics_from_cov(cov, self.n, self.dim)


# =============================================================================
# Reconstruction utilities
# =============================================================================

def normalize_for_metrics(x, rescale_from_neg11=False):
    """
    Bring an input or reconstruction to [0, 1] for image metrics.

    `rescale_from_neg11`: if True, assume `x` is in [-1, 1] and apply
    `(x + 1) / 2` before clamping. If False (default), just clamp.

    This is a user-controlled switch — NOT auto-detected per batch. An
    earlier version probed `x.amin() < -0.01` and rescaled adaptively, which
    fired inconsistently across batches (the recon's floor floats around
    -0.003 ± noise) and silently corrupted PSNR / SSIM / LPIPS / rFID. The
    fix is to pick one convention per evaluation run and stick with it.
    """
    if rescale_from_neg11:
        x = (x + 1) * 0.5
    return x.clamp(0.0, 1.0)


def to_rgb(x):
    """Force a (B, C, H, W) tensor to 3 channels by replicating the grayscale
    channel. Used for LPIPS and PNG dumps for rFID."""
    if x.shape[1] == 3:
        return x
    if x.shape[1] == 1:
        return x.expand(-1, 3, -1, -1).contiguous()
    raise RuntimeError(f"evaluation: unexpected channel count {x.shape[1]}; "
                       "expected 1 or 3.")


def save_image_batch(images_01, out_dir, start_idx):
    """
    Dump a batch of [0, 1] images to `out_dir/<8-digit-idx>.png` for
    consumption by torch_fidelity. Images are forced to 3-channel RGB
    (Inception expects RGB) and quantized to uint8. The directory must
    already exist.

    Mirrors how `VAEReconstruction.save_evaluation_outputs` →
    `ToDisk.vid_or_image_to_png` writes its PNGs (torchvision.io.write_png
    on a uint8 CHW tensor), but flat-numbered so torch_fidelity sees the
    whole dataset as one directory.
    """
    imgs = to_rgb(images_01).clamp(0, 1)
    imgs_u8 = (imgs * 255).round().to(torch.uint8).cpu()
    n = imgs_u8.shape[0]
    for i in range(n):
        path = os.path.join(out_dir, f"{start_idx + i:08d}.png")
        write_png(imgs_u8[i], path)
    return start_idx + n


# NOTE: FID is not computed here. evaluate_tokenizer.py only dumps the per-image
# sample/ (reconstructions) and optional reference/ (real GT) PNGs; all FID and
# MedFID scores are computed offline by calculate_fids.py over those PNG dirs.


# =============================================================================
# Per-dataset evaluation
# =============================================================================

@torch.no_grad()
def evaluate_dataset(model, loader, *, device, dataset_name,
                     fid_real_dir, fid_recon_dir,
                     rescale_from_neg11=False,
                     write_references=True,
                     autocast_dtype=torch.bfloat16):
    """
    Run the full metric panel on one dataset. Returns a flat row dict suitable
    for CSV plus the geometry sub-dicts.

    - Reconstruction metrics PSNR/SSIM/LPIPS/MSE/MAE come from torchmetrics +
      direct tensor sums during the loop.
    - FID/MedFID are NOT computed here anymore. This function only dumps the
      per-image PNGs: reconstructions to ``fid_recon_dir`` (the persistent
      sample/ output, always written) and, when ``write_references`` is True,
      the real GT to ``fid_real_dir`` (reference/). Compute FID/MedFID offline
      from these directories with calculate_fids.py.
    - Latent geometry is accumulated online over *every* spatial position in
      the split — no subsampling, O(D²) memory.
    """
    if write_references:
        os.makedirs(fid_real_dir, exist_ok=True)
    os.makedirs(fid_recon_dir, exist_ok=True)

    # Discrete VQ models expose `model.quantizer`; continuous AEKL/medVAE
    # don't. We compute codebook metrics only when a quantizer is present.
    quantizer = getattr(model, "quantizer", None)
    if quantizer is not None:
        q_cls = type(quantizer).__name__
        K = infer_codebook_size(quantizer)
    else:
        q_cls = None
        K = 0
        print(f"[eval] {dataset_name}: no quantizer on model "
              f"(type={type(model).__name__}); skipping codebook metrics.")

    # Per-image metric arrays — accumulated as a list of 1-D tensors and
    # concatenated at the end so we can report both mean *and* std across
    # subjects per dataset. PSNR/MSE/MAE are vectorized over the batch axis;
    # SSIM uses the torchmetrics functional version with reduction='none';
    # LPIPS goes through the raw `lpips` network so we get per-image values
    # (the torchmetrics class reduces internally).
    lpips_net = lpips_pkg.LPIPS(net="alex", verbose=False).to(device).eval()
    psnr_per, ssim_per, lpips_per, mse_per, mae_per = [], [], [], [], []

    counts = torch.zeros(K, dtype=torch.float64) if K > 0 else None
    latent_cov = OnlineCovariance()
    image_idx = 0

    t0 = time.time()
    for batch in tqdm(loader, leave=False, dynamic_ncols=True, desc=dataset_name):
        img = batch[0] if isinstance(batch, (list, tuple)) else batch["image"]
        img = img.to(device, non_blocking=True)

        # ---- encode + decode (single forward) -------------------------------
        # For discrete VQ models, `encode(x)` returns the post-quant latent
        # and we also call `encode_to_prequant(x)` for the geometry analysis
        # (so we get the encoder's actual output distribution, not the
        # already-snapped-to-codes version). Continuous models don't have an
        # `encode_to_prequant`; their `encode` IS the latent, so we reuse it.
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            enc_out = model.encode(img)
            quant = enc_out[0]
            recon = model.decode(quant)
            if hasattr(model, "encode_to_prequant"):
                pre_quant = model.encode_to_prequant(img)[0]
            else:
                pre_quant = quant

        # ---- reconstruction metrics in fp32, normalized to [0, 1] -----------
        recon_n = normalize_for_metrics(recon.float(), rescale_from_neg11)
        img_n = normalize_for_metrics(img.float(), rescale_from_neg11)

        # Per-image MSE / MAE / PSNR (vectorized over batch axis).
        diff = recon_n - img_n
        mse_batch = (diff * diff).flatten(1).mean(dim=1)
        mae_batch = diff.abs().flatten(1).mean(dim=1)
        psnr_batch = -10.0 * torch.log10(mse_batch.clamp_min(1e-12))

        # Per-image SSIM (torchmetrics functional, reduction='none').
        ssim_batch = ssim_fn(recon_n, img_n, data_range=1.0, reduction="none")

        # Per-image LPIPS via the raw network (the class reduces internally).
        recon_rgb = to_rgb(recon_n)
        img_rgb = to_rgb(img_n)
        lpips_batch = lpips_net(recon_rgb, img_rgb, normalize=True).flatten()

        mse_per.append(mse_batch.detach().cpu())
        mae_per.append(mae_batch.detach().cpu())
        psnr_per.append(psnr_batch.detach().cpu())
        ssim_per.append(ssim_batch.detach().cpu())
        lpips_per.append(lpips_batch.detach().cpu())

        # ---- dump real + recon PNGs (real only when caller asked) ----------
        if write_references:
            save_image_batch(img_n, fid_real_dir, image_idx)
        image_idx = save_image_batch(recon_n, fid_recon_dir, image_idx)

        # ---- codebook accumulation (discrete models only) -------------------
        if counts is not None:
            idx = extract_indices_from_encode(enc_out, q_cls)
            if idx is not None:
                flat = flatten_indices(idx, quantizer, K)
                counts += torch.bincount(flat, minlength=K).double()

        # ---- streaming latent geometry over EVERY position ------------------
        pre_f = pre_quant.float()
        if pre_f.dim() >= 3:
            b, c = pre_f.shape[0], pre_f.shape[1]
            zp = pre_f.reshape(b, c, -1).permute(0, 2, 1).reshape(-1, c).contiguous()
            latent_cov.update_batch(zp)

    elapsed_encode = time.time() - t0

    # ---- finalize -----------------------------------------------------------
    def _mean_std(per_list):
        """Pool a list of 1-D per-image tensors and return (mean, std).
        Sample std (ddof=1); returns 0.0 if there's only one image."""
        if not per_list:
            return float("nan"), float("nan")
        x = torch.cat([t.flatten() for t in per_list]).double()
        n = x.numel()
        if n == 0:
            return float("nan"), float("nan")
        mean = float(x.mean().item())
        std = float(x.std(unbiased=True).item()) if n > 1 else 0.0
        return mean, std

    psnr_v, psnr_sd = _mean_std(psnr_per)
    ssim_v, ssim_sd = _mean_std(ssim_per)
    lpips_v, lpips_sd = _mean_std(lpips_per)
    mse_v, mse_sd = _mean_std(mse_per)
    mae_v, mae_sd = _mean_std(mae_per)
    n_images_metric = int(sum(t.numel() for t in mse_per))

    if counts is not None:
        cb_metrics = codebook_metrics_from_counts(counts, K)
    else:
        # Continuous model — no codebook to score. Keep the keys in the row
        # so the CSV header is stable across the sweep, but leave them empty.
        cb_metrics = {
            "codebook_usage": "", "codebook_entropy": "",
            "codebook_utility": "", "codebook_used": "",
            "codebook_size": "", "codebook_total_assignments": "",
        }
    geo_lat = latent_cov.metrics()
    elapsed = elapsed_encode

    if counts is not None and quantizer is not None:
        used_mask = (counts > 0).numpy()
        try:
            cb_used = extract_used_codebook_matrix(quantizer, used_mask)
            geo_cb = latent_geometry(cb_used)
        except RuntimeError as e:
            print(f"  [WARN] codebook geometry skipped: {e}")
            geo_cb = _empty_geo()
    else:
        geo_cb = _empty_geo()

    # Drop the LPIPS network for the next dataset; the metric arrays are
    # already on CPU.
    del lpips_net
    if device.type == "cuda":
        torch.cuda.empty_cache()

    row = {
        "dataset": dataset_name,
        # mean ± std reported across subjects (one value per test image)
        "psnr": psnr_v, "psnr_std": psnr_sd,
        "ssim": ssim_v, "ssim_std": ssim_sd,
        "lpips": lpips_v, "lpips_std": lpips_sd,
        "mse": mse_v, "mse_std": mse_sd,
        "mae": mae_v, "mae_std": mae_sd,
        **cb_metrics,
        "latent_stable_rank": geo_lat["stable_rank"],
        "latent_effective_rank": geo_lat["effective_rank"],
        "latent_anisotropy": geo_lat["anisotropy"],
        "codebook_stable_rank": geo_cb["stable_rank"],
        "codebook_effective_rank": geo_cb["effective_rank"],
        "codebook_anisotropy": geo_cb["anisotropy"],
        "n_subjects": int(n_images_metric),
        "n_images_dumped": int(image_idx),
        "elapsed_encode_sec": float(elapsed_encode),
        "elapsed_sec": float(elapsed),
    }
    extra = {
        "latent_geometry": geo_lat,
        "codebook_geometry": geo_cb,
        "quantizer_class": q_cls,
        "fid_real_dir": fid_real_dir,
        "fid_recon_dir": fid_recon_dir,
        # Raw count buffer for the across-datasets pool in main(). Caller is
        # expected to pop this before any JSON serialization (it's a tensor).
        # None for continuous models — main() skips pooled accumulation then.
        "codebook_counts": counts,
        "codebook_K": int(K) if K > 0 else None,
    }
    return row, extra


# =============================================================================
# Top-level driver
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        help="medlat model names (e.g. discrete.vq.f8_d4_e16384).")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Subset of MedMNIST 2D datasets; default = all 12.")
    parser.add_argument("--base_dir", type=str, default=paths.BASE_DIR,
                        help="Folder containing outputs/<project>/ ($RMT_BASE_DIR).")
    parser.add_argument("--project", type=str, default=paths.PROJECT)
    parser.add_argument("--data_root", type=str, default=paths.MEDMNIST_ROOT,
                        help="MedMNIST+ .npz folder ($MEDMNIST_ROOT).")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--ckpt", type=int, default=None,
                        help="Checkpoint step to load (default: latest).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--recon_dir", type=str, default=None,
                        help="Override the root location of the per-dataset "
                             "reconstruction PNG dumps. Default: "
                             "<base_dir>/outputs/<project>/<model>/reconstruction/. "
                             "Put this on fast scratch storage when you have it. "
                             "Per-dataset layout under the root is "
                             "<dataset>/sample/<8digit>.png (reconstructions, "
                             "always kept) and optionally "
                             "<dataset>/reference/<8digit>.png (real GT, "
                             "controlled by --save_references).")
    parser.add_argument("--save_references", action="store_true",
                        help="Write (and keep) the per-dataset reference/ folder "
                             "of real GT PNGs. REQUIRED if you want to compute "
                             "reconstruction FID/MedFID offline with "
                             "calculate_fids.py (it needs both sample/ and "
                             "reference/). Off by default to avoid writing the "
                             "identical reference subset once per tokenizer; the "
                             "reconstructions in sample/ are always kept.")
    parser.add_argument("--normalize", action="store_true",
                        help="Rescale inputs and reconstructions from [-1, 1] "
                             "to [0, 1] before computing image metrics. Off by "
                             "default — only enable if your model was trained "
                             "with images in [-1, 1] (i.e. trainer.normalize: "
                             "True in the YAML).")
    args = parser.parse_args()

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)
    datasets = list(args.datasets) if args.datasets else list(ALL_2D)

    print(f"[eval] models ({len(args.models)}): {args.models}")
    print(f"[eval] datasets ({len(datasets)}): {datasets}")

    for model_name in args.models:
        print(f"\n========== MODEL: {model_name} ==========")

        # Load first so we can place evaluation outputs as a sibling of
        # train/ and probing/ inside the model's existing output directory,
        # i.e. <base_dir>/outputs/<project>/<model>/evaluation/.
        info = load_first_stage(
            model_name, base_dir=args.base_dir, project=args.project,
            ckpt=args.ckpt, img_size=args.image_size, device=device,
        )
        model = info["model"]
        eval_dir = os.path.join(info["model_dir"], "evaluation")
        os.makedirs(eval_dir, exist_ok=True)
        csv_path = os.path.join(eval_dir, "results.csv")
        json_path = os.path.join(eval_dir, "results.json")
        recon_root = args.recon_dir or os.path.join(info["model_dir"], "reconstruction")
        print(f"[eval] eval dir: {eval_dir}")
        if info["family"] != "discrete":
            print(f"[eval] {model_name}: family={info['family']} has no codebook; "
                  "running with reconstruction + latent metrics only.")
        model_rows = []
        model_json = {
            "model": model_name,
            "family": info["family"],
            "ckpt_step": info["ckpt_step"],
            "args": vars(args),
            "datasets": {},
            "pooled_codebook": None,
        }
        # Running across-datasets counts buffer for this model. Lazy-init on
        # the first dataset, summed in place after each one; used to compute
        # the "_all_" pooled codebook row at the end of the model's loop.
        pooled_counts = None
        pooled_K = None

        for ds_name in datasets:
            print(f"\n--- {model_name} :: {ds_name} ({args.split}) ---")
            ref_dir = os.path.join(recon_root, ds_name, "reference")
            sample_dir = os.path.join(recon_root, ds_name, "sample")
            try:
                loaders, _ = build_probe_loaders(
                    ds_name, root=args.data_root, batch_size=args.bs,
                    num_workers=args.num_workers,
                    image_size=args.image_size, download=False,
                )
                # References are written only when persisted (FID/MedFID are
                # computed offline by calculate_fids.py, which needs them).
                write_refs = args.save_references
                row, extra = evaluate_dataset(
                    model, loaders[args.split], device=device,
                    dataset_name=ds_name,
                    fid_real_dir=ref_dir,
                    fid_recon_dir=sample_dir,
                    rescale_from_neg11=args.normalize,
                    write_references=write_refs,
                )
            except Exception as e:
                print(f"[eval] FAIL {model_name}::{ds_name}: {e}")
                import traceback; traceback.print_exc()
                continue

            # Pop the raw counts tensor before JSON serialization — it's not
            # JSON-safe and we only need it for cross-dataset pooling.
            cb_counts = extra.pop("codebook_counts", None)
            cb_K = extra.pop("codebook_K", None)
            if cb_counts is not None and cb_K is not None:
                if pooled_counts is None:
                    pooled_counts = torch.zeros_like(cb_counts)
                    pooled_K = cb_K
                if cb_K != pooled_K:
                    print(f"[eval] WARN K mismatch across datasets "
                          f"({pooled_K} -> {cb_K}); skipping pooled accumulation.")
                else:
                    pooled_counts += cb_counts

            model_rows.append(row)
            model_json["datasets"][ds_name] = {**row, **extra}

            print(f"  psnr={row['psnr']:.3f}±{row['psnr_std']:.3f}  "
                  f"ssim={row['ssim']:.4f}±{row['ssim_std']:.4f}  "
                  f"lpips={row['lpips']:.4f}±{row['lpips_std']:.4f}  "
                  f"mse={row['mse']:.5f}±{row['mse_std']:.5f}  "
                  f"mae={row['mae']:.5f}±{row['mae_std']:.5f}")
            if isinstance(row.get("codebook_usage"), float):
                print(f"  codebook: usage={row['codebook_usage']:.4f} "
                      f"({row['codebook_used']}/{row['codebook_size']})  "
                      f"entropy={row['codebook_entropy']:.4f}  "
                      f"utility={row['codebook_utility']:.4f}")
            else:
                print(f"  codebook: (continuous model, no codebook)")
            print(f"  latents : stable_rank={row['latent_stable_rank']:.2f}  "
                  f"eff_rank={row['latent_effective_rank']:.2f}  "
                  f"anisotropy={row['latent_anisotropy']:.3g}")
            print(f"  used cb : stable_rank={row['codebook_stable_rank']:.2f}  "
                  f"eff_rank={row['codebook_effective_rank']:.2f}  "
                  f"anisotropy={row['codebook_anisotropy']:.3g}")
            print(f"  ({row['elapsed_sec']:.1f}s)")

            # write incrementally so an interrupted run leaves usable output
            _write_csv(csv_path, model_rows)
            with open(json_path, "w") as f:
                json.dump(model_json, f, indent=2, default=str)

            # Reconstructions are valuable outputs of the eval and always
            # persisted (this is the sample/ dir under reconstruction/).
            # References are bulky and identical across tokenizers, so unless
            # --save_references was set the dir was never written — and we drop
            # any stray one here. With --save_references on, references are kept
            # so calculate_fids.py can compute reconstruction FID/MedFID.
            if not args.save_references:
                try:
                    shutil.rmtree(ref_dir)
                except FileNotFoundError:
                    pass
                except Exception as e:
                    print(f"[eval] WARN couldn't remove {ref_dir}: {e}")

        # ---- per-model pooled codebook row (across ALL datasets) ----
        # Tells you whether the model has a single "active subset" of codes
        # shared across modalities or whether each dataset activates a
        # different slice of the codebook. Compare with the per-dataset
        # codebook_used numbers above.
        if pooled_counts is not None and pooled_K is not None:
            pooled_metrics = codebook_metrics_from_counts(pooled_counts, pooled_K)
            pooled_row = {
                "dataset": "_all_",
                **pooled_metrics,
            }
            model_rows.append(pooled_row)
            model_json["pooled_codebook"] = pooled_metrics
            print(f"\n[eval] {model_name} POOLED across {len(datasets)} datasets: "
                  f"usage={pooled_metrics['codebook_usage']:.4f} "
                  f"({pooled_metrics['codebook_used']}/{pooled_metrics['codebook_size']})  "
                  f"entropy={pooled_metrics['codebook_entropy']:.4f}  "
                  f"utility={pooled_metrics['codebook_utility']:.4f}")
            _write_csv(csv_path, model_rows)
            with open(json_path, "w") as f:
                json.dump(model_json, f, indent=2, default=str)

        print(f"[eval] -> {csv_path}")
        del model, info
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n[eval] DONE.")


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    seen = set(fieldnames)
    for r in rows[1:]:
        for k in r:
            if k not in seen:
                fieldnames.append(k); seen.add(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
