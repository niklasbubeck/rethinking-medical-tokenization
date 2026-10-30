from sklearn import metrics
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import os 
import torch.nn.functional as F
import torch_fidelity
import itertools
from monai.inferers import sliding_window_inference
from ..utils import *
from ..metrics.metrics import CompositeMetrics

from .base_evaluator import EvaluationTask

class VAEReconstruction(EvaluationTask):
    def __init__(self, name, metrics, input_key='volume', save_as=['gif', 'nii.gz'], n_samples=None):
        super().__init__(name, metrics, save_as)
        self.name = name
        self.n_samples = n_samples 
        self.input_key = input_key
    @torch.no_grad()
    def process_batch(self, model, batch, output_dir=None):
        model = self.unwrap_model(model)
        gts = batch[self.input_key]

        outputs, *_ = model(gts)

        return gts, outputs, torch.ones_like(gts)




class VAESlidingWindowReconstruction(VAEReconstruction):
    def __init__(self, name, metrics, input_key='volume', save_as=['gif', 'nii.gz'], n_samples=None):
        super().__init__(name, metrics, save_as)
        self.name = name
        self.n_samples = n_samples 
        self.input_key = input_key

    def predictor(self, x):
        reconstruction, *_ = self.model(x)
        return reconstruction

    @torch.no_grad()
    def process_batch(self, model, batch, output_dir=None):
        self.model = self.unwrap_model(model)
        gts = batch[self.input_key]

        outputs = sliding_window_inference(
            inputs=gts,    # tensor (1, 1, 512, 512, 512)
            roi_size=(64, 64, 64),
            sw_batch_size=1,
            predictor=self.predictor,
            overlap=0.0,
        )
        # z = model.encode_sliding_window(gts)
        # outputs = model.decode_sliding_window(z)


        return gts, outputs, torch.ones_like(gts)

class MAISIReconstruction(EvaluationTask):
    def __init__(self, name, metrics, input_key='volume', save_as=['gif', 'nii.gz'], n_samples=None):
        super().__init__(name, metrics, save_as)
        self.name = name
        self.n_samples = n_samples 
        self.input_key = input_key
    @torch.no_grad()
    def process_batch(self, model, batch, output_dir=None):
        model = self.unwrap_model(model)
        gts = batch[self.input_key]

        with torch.autocast("cuda", enabled=True):
            latent = model.sampling(*model.encode(gts))
            outputs = model.decode(latent)

        return gts, outputs, torch.ones_like(gts)
 
class VAEPatchReconstruction(VAEReconstruction):
    def __init__(self, name, metrics, input_key='volume', save_as=['gif', 'nii.gz'], n_samples=None):
        super().__init__(name, metrics, save_as)
        self.name = name
        self.n_samples = n_samples 
        self.input_key = input_key

    def process_batch(
        self,
        model,
        batch,
        output_dir=None,
        num_splits_per_dim=4,  # e.g., 2 → 8 patches, 4 → 64 patches
        pad_if_needed=True,
    ):
        """
        Split 3D volume into num_splits_per_dim³ subvolumes, run each patch sequentially
        through the model to save memory, and reassemble the final output.
        """
        model = self.unwrap_model(model)
        gts = batch[self.input_key]  # shape: (B, C, D, H, W)
        B, C, D, H, W = gts.shape
        device = gts.device

        # ---- Optionally pad so dimensions are divisible by num_splits_per_dim ----
        pad_D = pad_H = pad_W = 0
        if pad_if_needed:
            pad_D = (num_splits_per_dim - D % num_splits_per_dim) % num_splits_per_dim
            pad_H = (num_splits_per_dim - H % num_splits_per_dim) % num_splits_per_dim
            pad_W = (num_splits_per_dim - W % num_splits_per_dim) % num_splits_per_dim
            if pad_D or pad_H or pad_W:
                gts = F.pad(gts, (0, pad_W, 0, pad_H, 0, pad_D))
                D, H, W = gts.shape[2:]

        # ---- Compute subvolume sizes ----
        d_size = D // num_splits_per_dim
        h_size = H // num_splits_per_dim
        w_size = W // num_splits_per_dim

        # ---- Prepare output tensor ----
        full_output = torch.zeros_like(gts, dtype=torch.float32, device=device)

        model.eval()
        with torch.no_grad():
            for dz, dy, dx in itertools.product(range(num_splits_per_dim), repeat=3):
                z0, y0, x0 = dz * d_size, dy * h_size, dx * w_size
                z1, y1, x1 = z0 + d_size, y0 + h_size, x0 + w_size

                # Extract patch
                patch = gts[:, :, z0:z1, y0:y1, x0:x1]

                # Forward pass (serially)
                outputs, *_ = model(patch)

                # Place back in correct position
                full_output[:, :, z0:z1, y0:y1, x0:x1] = outputs

                # Optional: free GPU memory after each patch
                del patch, outputs
                torch.cuda.empty_cache()

        # ---- Remove padding ----
        if pad_if_needed and (pad_D or pad_H or pad_W):
            full_output = full_output[:, :, 0:D-pad_D, 0:H-pad_H, 0:W-pad_W]
            gts = gts[:, :, 0:D-pad_D, 0:H-pad_H, 0:W-pad_W]

        # ---- Metric mask (optional) ----
        metric_mask = torch.ones_like(gts)

        return gts, full_output, metric_mask

class VAEKL3DReconstruction(VAEReconstruction):
    def __init__(self, name, metrics, input_key='volume', save_as=['gif', 'nii.gz'], n_samples=None):
        super().__init__(name, metrics, input_key, save_as, n_samples)

    def process_batch(self, model, batch, output_dir=None):
        model = self.unwrap_model(model)
        gts, labels = batch[self.input_key], batch['label']
        gts = torch.nn.functional.interpolate(gts, size=(64, 64, 64), mode='trilinear', align_corners=False)
        outputs, *_ = model(gts)
        return gts, outputs, torch.ones_like(gts)
    
class VQVAE3DReconstruction(VAEReconstruction):
    def __init__(self, name, metrics, save_as=['gif', 'nii.gz'], n_samples=None):
        super().__init__(name, metrics, save_as, n_samples)

    def process_batch(self, model, batch, output_dir=None):
        model = self.unwrap_model(model)
        gts, labels = batch[self.input_key], batch['label']
        outputs, *_ = model(gts)
        return gts, outputs, torch.ones_like(gts)


class VQUtilityEvaluation(VAEReconstruction):
    """
    Evaluates VQ model utility by measuring codebook usage, entropy, and uniformity.
    Extends VAEReconstruction to also compute reconstruction metrics.

    Codebook metrics are CUMULATIVE across the whole evaluation split, not
    per-batch averages. We maintain a running per-code counts buffer in
    `process_batch` and emit the final metrics once via
    `_compute_generation_metrics`, which `evaluate()` merges into the
    returned metrics dict at the end of the run.
    """
    def __init__(self, name, metrics, input_key='image', save_as=['png'], n_samples=None):
        super().__init__(name, metrics, input_key, save_as, n_samples)
        # Cumulative state — initialized on the first batch and reset at the
        # end of every evaluate() via _compute_generation_metrics, so the
        # same instance can be re-used across datasets.
        self._cum_counts = None              # 1-D float64 tensor of length K
        self._cum_codebook_size = None       # int
        self._cum_quantizer_class = None     # str (for sanity-checking)

    @torch.no_grad()
    def process_batch(self, model, batch, output_dir=None):
        model = self.unwrap_model(model)
        gts = batch[self.input_key]

        # Get reconstruction and quantized indices
        # Try different ways VQ models might return indices
        try:
            if hasattr(model, 'forward') and 'return_pred_indices' in model.forward.__code__.co_varnames:
                outputs, qloss, indices = model(gts, return_pred_indices=True)
            else:
                # Try encode method
                quant, emb_loss, info = model.encode(gts)
                if isinstance(info, tuple) and len(info) >= 3:
                    indices = info[2]  # Usually (perplexity, encodings, indices)
                elif isinstance(info, dict) and 'min_encoding_indices' in info:
                    indices = info['min_encoding_indices']
                else:
                    indices = None
                outputs = model.decode(quant)
        except Exception as e:
            print(f"Warning: Could not extract indices from model: {e}")
            outputs, *_ = model(gts)
            indices = None

        # Accumulate this batch's per-code counts into the cumulative buffer.
        # Cumulative codebook metrics are emitted at the end of evaluate()
        # by _compute_generation_metrics — NOT per batch.
        if indices is not None:
            self._accumulate_counts(model, indices)

        return gts, outputs, torch.ones_like(gts)
    
    @staticmethod
    def _resolve_quantizer(model):
        """
        Return the quantizer *module* of a medlat VQ model.

        IMPORTANT: medlat's VQModel exposes the quantizer module at
        `model.quantizer` and the method `def quantize(self, h)` at
        `model.quantize`. An earlier version of this code probed `model.quantize`
        — that's a bound method, so every `hasattr(model.quantize, 'n_e' / ...)`
        check silently failed and `codebook_size` ended up being inferred from
        `max_index + 1`, which is not the codebook size.
        """
        return getattr(model, 'quantizer', None)

    @staticmethod
    def _level_offsets(quantizer):
        """
        For a ResidualQuantizer, return per-level start offsets so concatenated
        indices live in a disjoint global address space. Each level's indices
        are independently in [0, n_e); without offsets, "code 0 in level 0"
        would be conflated with "code 0 in level 1".

        Returns None for non-residual quantizers, or for shared-codebook RQs
        where the same codebook is reused across levels (no offset needed).
        """
        if quantizer is None or not hasattr(quantizer, 'levels'):
            return None
        levels = list(quantizer.levels)
        if not levels:
            return None
        if getattr(quantizer, 'shared_codebook', False):
            return [0] * len(levels)
        sizes = [int(getattr(lv, 'n_e', 0)) for lv in levels]
        if any(s <= 0 for s in sizes):
            return None
        offsets = [0]
        for s in sizes[:-1]:
            offsets.append(offsets[-1] + s)
        return offsets

    @staticmethod
    def _infer_codebook_size(quantizer):
        """
        Resolve the true codebook size from a medlat quantizer module.

        Returns None if we can't be sure — callers MUST surface that to the
        user rather than silently fabricating a size from the indices.
        """
        if quantizer is None:
            return None
        # ResidualQuantizer: sum across levels (matches the offset scheme in
        # _level_offsets so that bincount addresses every (level, code) slot).
        if hasattr(quantizer, 'levels') and hasattr(quantizer, 'num_quantizers'):
            levels = list(quantizer.levels)
            if not levels:
                return None
            if getattr(quantizer, 'shared_codebook', False):
                per = int(getattr(levels[0], 'n_e', 0))
                return per or None
            sizes = [int(getattr(lv, 'n_e', 0)) for lv in levels]
            if any(s <= 0 for s in sizes):
                return None
            return sum(sizes)
        # Single-level path: prefer `n_e` (true on VQ / SimVQ / SimpleQINCo,
        # exposed as a property by LFQ / BSQ via `n_e -> codebook_size`).
        for attr in ('n_e', 'codebook_size'):
            val = getattr(quantizer, attr, None)
            if val is not None:
                return int(val)
        # Embedding-table fallback for future quantizers that expose one.
        emb = getattr(quantizer, 'embedding', None)
        if emb is not None:
            if hasattr(emb, 'num_embeddings'):
                return int(emb.num_embeddings)
            if hasattr(emb, 'weight') and hasattr(emb.weight, 'shape'):
                return int(emb.weight.shape[0])
        return None

    def _compute_codebook_utility(self, model, indices):
        """
        Compute codebook utility metrics from quantized indices.

        Args:
            model: The medlat VQ model. The quantizer module is read from
                   `model.quantizer` (NOT `model.quantize`, which is a method).
            indices: Tensor of quantized indices, or a list of per-level
                     tensors for ResidualQuantizer. Shapes may vary:
                       - (B, H, W) / (B, D, H, W) / (B, N) for tensors
                       - list of any of the above (one per RQ level)

        Returns:
            dict with utility metrics.

        Raises:
            RuntimeError: if the true codebook size can't be resolved or if
                          the supplied indices are inconsistent with the
                          quantizer's declared size. We raise rather than
                          silently fall back, because the prior fallback
                          ("infer codebook_size from max_index + 1") produced
                          plausible-looking but wrong utilization numbers.
        """
        quantizer = self._resolve_quantizer(model)
        if quantizer is None:
            raise RuntimeError(
                "VQUtilityEvaluation: could not locate the quantizer module on "
                f"the model (type={type(model).__name__}). Expected attribute "
                "`model.quantizer`. Note: `model.quantize` is a *method*, not "
                "the module — checks against it silently fail."
            )

        # Normalize indices to a 1-D long tensor in a single global address
        # space (offsetting per level for non-shared ResidualQuantizers so the
        # union of codes is what's counted, not the per-level [0, n_e) slot).
        if isinstance(indices, (list, tuple)):
            offsets = self._level_offsets(quantizer)
            if offsets is None:
                raise RuntimeError(
                    "VQUtilityEvaluation: received a list of per-level indices "
                    f"but quantizer={type(quantizer).__name__} doesn't expose "
                    "`levels` / per-level `n_e` — can't compute per-level "
                    "offsets. Pass a single tensor of indices or use a "
                    "ResidualQuantizer-compatible quantizer."
                )
            if len(offsets) != len(indices):
                raise RuntimeError(
                    f"VQUtilityEvaluation: got {len(indices)} per-level index "
                    f"tensors but quantizer has {len(offsets)} levels."
                )
            shifted = [i.reshape(-1).long() + off for i, off in zip(indices, offsets)]
            indices_flat = torch.cat(shifted, dim=0)
        elif torch.is_tensor(indices):
            indices_flat = indices.reshape(-1).long()
        else:
            raise RuntimeError(
                "VQUtilityEvaluation: indices must be a torch.Tensor or a list "
                f"of torch.Tensor (got {type(indices).__name__})."
            )
        indices_flat = indices_flat.cpu()
        # ResidualQuantizer uses -1 as a dropped-level sentinel; ignore those.
        indices_flat = indices_flat[indices_flat >= 0]

        codebook_size = self._infer_codebook_size(quantizer)
        if codebook_size is None or codebook_size <= 0:
            raise RuntimeError(
                "VQUtilityEvaluation: could not resolve codebook size from "
                f"quantizer={type(quantizer).__name__}. Tried `n_e`, "
                "`codebook_size`, `embedding.num_embeddings`. If this is a new "
                "quantizer family, extend `_infer_codebook_size`."
            )

        if indices_flat.numel() > 0:
            max_idx = int(indices_flat.max().item())
            if max_idx >= codebook_size:
                raise RuntimeError(
                    "VQUtilityEvaluation: observed index "
                    f"{max_idx} but resolved codebook_size={codebook_size}. "
                    "The quantizer's declared size disagrees with what came "
                    "out of encode() — refusing to silently produce wrong "
                    "utilization. Check `_resolve_quantizer` / "
                    "`_infer_codebook_size` for this quantizer family."
                )

        # --- usage / entropy / utilization (math unchanged) ---
        unique_indices = torch.unique(indices_flat)
        num_used_codes = len(unique_indices)
        codebook_usage = num_used_codes / codebook_size

        counts = torch.bincount(indices_flat, minlength=codebook_size).float()
        probs = counts / (counts.sum() + 1e-10)
        entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
        max_entropy = np.log2(codebook_size) if codebook_size > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        mean_usage = probs.mean().item()
        std_usage = probs.std().item()
        cv = std_usage / (mean_usage + 1e-10) if mean_usage > 0 else 1.0
        utilization = 1.0 / (1.0 + cv)

        active_codes = (counts > 0).sum().item()
        avg_usage_per_code = counts[counts > 0].mean().item() if active_codes > 0 else 0.0
        max_usage = counts.max().item()
        min_usage = counts[counts > 0].min().item() if active_codes > 0 else 0.0

        return {
            'codebook_usage': codebook_usage,           # fraction of codebook ever selected
            'codebook_entropy': normalized_entropy,     # H(p) / log2(K)
            'codebook_utilization': utilization,        # 1 / (1 + CV) — uniformity
            'num_used_codes': num_used_codes,
            'codebook_size': codebook_size,
            'active_codes': active_codes,
            'avg_usage_per_code': avg_usage_per_code,
            'max_usage': max_usage,
            'min_usage': min_usage,
            'raw_entropy': entropy,
        }
    
    def _accumulate_counts(self, model, indices):
        """
        Add this batch's per-code counts into the running cumulative buffer
        `self._cum_counts`. Reuses the same index-resolution logic as
        `_compute_codebook_utility` (quantizer module lookup, per-level
        offsets for ResidualQuantizer, sentinel filtering, bounds check).

        Raises the same RuntimeErrors as `_compute_codebook_utility` if the
        quantizer module / codebook size / index layout can't be resolved.
        """
        quantizer = self._resolve_quantizer(model)
        if quantizer is None:
            raise RuntimeError(
                "VQUtilityEvaluation: could not locate the quantizer module on "
                f"the model (type={type(model).__name__}). Expected attribute "
                "`model.quantizer`. Note: `model.quantize` is a *method*, not "
                "the module — checks against it silently fail."
            )

        if isinstance(indices, (list, tuple)):
            offsets = self._level_offsets(quantizer)
            if offsets is None:
                raise RuntimeError(
                    "VQUtilityEvaluation: received a list of per-level indices "
                    f"but quantizer={type(quantizer).__name__} doesn't expose "
                    "`levels` / per-level `n_e` — can't compute per-level offsets."
                )
            if len(offsets) != len(indices):
                raise RuntimeError(
                    f"VQUtilityEvaluation: got {len(indices)} per-level index "
                    f"tensors but quantizer has {len(offsets)} levels."
                )
            shifted = [i.reshape(-1).long() + off for i, off in zip(indices, offsets)]
            flat = torch.cat(shifted, dim=0)
        elif torch.is_tensor(indices):
            flat = indices.reshape(-1).long()
        else:
            raise RuntimeError(
                "VQUtilityEvaluation: indices must be a torch.Tensor or a list "
                f"of torch.Tensor (got {type(indices).__name__})."
            )
        flat = flat.cpu()
        flat = flat[flat >= 0]   # strip ResidualQuantizer's -1 dropout sentinel

        K = self._infer_codebook_size(quantizer)
        if K is None or K <= 0:
            raise RuntimeError(
                "VQUtilityEvaluation: could not resolve codebook size from "
                f"quantizer={type(quantizer).__name__}. Tried `n_e`, "
                "`codebook_size`, `embedding.num_embeddings`."
            )
        if flat.numel() > 0:
            max_idx = int(flat.max().item())
            if max_idx >= K:
                raise RuntimeError(
                    "VQUtilityEvaluation: observed index "
                    f"{max_idx} but resolved codebook_size={K}. The quantizer's "
                    "declared size disagrees with what came out of encode()."
                )

        # Lazy-init the cumulative buffer on the first batch. Sanity-check
        # that K and quantizer class stay the same across batches.
        q_cls = type(quantizer).__name__
        if self._cum_counts is None:
            self._cum_codebook_size = K
            self._cum_quantizer_class = q_cls
            self._cum_counts = torch.zeros(K, dtype=torch.float64)
        else:
            if K != self._cum_codebook_size or q_cls != self._cum_quantizer_class:
                raise RuntimeError(
                    "VQUtilityEvaluation: codebook size or quantizer type "
                    "changed mid-evaluation "
                    f"({self._cum_quantizer_class}/K={self._cum_codebook_size} "
                    f"-> {q_cls}/K={K}). Call _reset_cumulative() between models."
                )

        batch_counts = torch.bincount(flat, minlength=K).double()
        self._cum_counts += batch_counts

    def _reset_cumulative(self):
        """Wipe cumulative state. Called at the end of `_compute_generation_metrics`
        so the same instance can be re-used for the next evaluation."""
        self._cum_counts = None
        self._cum_codebook_size = None
        self._cum_quantizer_class = None

    def _compute_generation_metrics(self):
        """
        Emit cumulative codebook metrics from `self._cum_counts` and merge
        them into the dict that `evaluate()` returns as `mean_metrics`.

        Called once at the end of every evaluation run (see `evaluate()` in
        `base_evaluator.py`). The reported numbers are computed from per-code
        counts pooled across ALL batches — they are NOT per-batch averages.
        """
        out = super()._compute_generation_metrics()

        if self._cum_counts is None:
            return out

        K = self._cum_codebook_size
        counts = self._cum_counts
        total = float(counts.sum().item())
        if total <= 0:
            self._reset_cumulative()
            return out

        n_used = int((counts > 0).sum().item())
        codebook_usage = n_used / K                          # fraction ever selected

        probs = counts / total
        nz = probs > 0
        entropy = float(-(probs[nz] * torch.log2(probs[nz])).sum().item())
        max_entropy = float(np.log2(K)) if K > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        mean = float(probs.mean().item())
        std = float(probs.std().item())
        cv = std / (mean + 1e-10) if mean > 0 else 1.0
        utilization = 1.0 / (1.0 + cv)                       # 1 / (1 + CV)

        out.update({
            "codebook_usage": codebook_usage,
            "codebook_entropy": normalized_entropy,
            "codebook_utilization": utilization,
            "codebook_used_codes": n_used,
            "codebook_size": int(K),
            "codebook_total_assignments": int(total),
        })

        self._reset_cumulative()
        return out