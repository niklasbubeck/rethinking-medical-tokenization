import torch
import torch.nn as nn


def _to_features(z, pooling):
    """
    Pool a latent of arbitrary shape into a 2D (B, F) feature matrix.

    Accepts:
        (B, C, H, W)         -> mean over H,W or flatten
        (B, C, D, H, W)      -> mean over D,H,W or flatten
        (B, C, 1, N) / (B, N, D)-style token tensors -> mean over non-channel dims or flatten / cls

    pooling:
        "mean":    average over every non-batch, non-channel-ish dim.
                   We treat the largest of the trailing dims as 'sequence/spatial' and dim=1 as feature.
                   For 4D (B,C,1,N) and (B,C,H,W) this collapses (1,N) or (H,W); both yield (B,C).
        "flatten": reshape the latent to (B, -1).
        "cls":     take index 0 along the largest non-batch dim that is > 1
                   (intended for token tensors with a CLS-like first token).
    """
    if z.dim() == 2:
        return z

    if pooling == "flatten":
        return z.reshape(z.shape[0], -1)

    if pooling == "cls":
        # find first non-batch dim with size > 1 that isn't dim=1 (channel)
        # token tensors here are typically (B, C, 1, N) or (B, N, D)
        # we take index 0 along dim=-1 if last dim is large, else dim=2
        last = z.shape[-1]
        if z.dim() >= 3 and last > 1:
            cls = z.index_select(-1, torch.tensor([0], device=z.device)).squeeze(-1)
        else:
            cls = z.index_select(2, torch.tensor([0], device=z.device)).squeeze(2)
        # If still >2D, mean-pool the rest
        if cls.dim() > 2:
            reduce_dims = tuple(range(2, cls.dim()))
            cls = cls.mean(dim=reduce_dims)
        return cls

    # default: mean over all non-batch, non-feature dims
    # convention: dim 1 is the feature dim for (B, C, ...) and (B, C, 1, N).
    # For (B, N, D) callers, we pre-permute to (B, D, N) before calling _to_features.
    if z.dim() >= 3:
        reduce_dims = tuple(range(2, z.dim()))
        return z.mean(dim=reduce_dims)
    return z


def _normalize_to_BCdots(z):
    """
    Best-effort: ensure feature dim is at axis 1.

    Heuristic: if z has shape (B, A, B') with A > B' and A > 64 (e.g. (B, N=128, D=12)),
    it's probably (B, tokens, dim) -> permute to (B, dim, tokens).
    Conservative: only permute 3D tensors when this clearly looks like tokens.
    """
    if z.dim() == 3:
        B, a, b = z.shape
        # token convention: middle dim is sequence (typically much larger than channel dim)
        if a > b and a >= 16:
            z = z.transpose(1, 2).contiguous()  # -> (B, b, a)
    return z


class FeatureExtractor(nn.Module):
    """
    Wraps a medlat first-stage model and extracts a fixed-length feature vector per image.

    Supports three families (auto-detected by `family` attribute on the model_info dict, or by
    presence of `encode_to_prequant`):

      - continuous.*     -> `encode(x)[0]` is a continuous latent (B, C, H, W)
      - discrete.*  (VQ) -> `encode(x)[0]` is the quantized latent;
                            `encode_to_prequant(x)` returns the pre-quant continuous latent.
                            extract_pair() returns BOTH (pre, post) from one encoder pass to
                            avoid double-running the encoder when probing both variants.
      - token.*          -> `encode(x)[0]` is a token tensor like (B, C, 1, N) or (B, N, D).

    The default pooling is global mean. The output is always (B, F) float32 on the same device.
    """

    def __init__(self, model, family, pooling="mean", autocast_dtype=torch.bfloat16):
        super().__init__()
        self.model = model
        self.family = family
        self.pooling = pooling
        self.autocast_dtype = autocast_dtype
        self.is_vq = family == "discrete" and hasattr(model, "encode_to_prequant")

    @property
    def variants(self):
        """Variant names this extractor produces. VQ models produce two; others produce one."""
        return ["pre", "post"] if self.is_vq else ["default"]

    @torch.no_grad()
    def _encode_raw(self, x):
        """Run the encoder, return whatever it returns (tuple expected)."""
        device_type = x.device.type
        if device_type == "cuda":
            with torch.autocast(device_type=device_type, dtype=self.autocast_dtype):
                out = self.model.encode(x)
        else:
            out = self.model.encode(x)
        return out

    @torch.no_grad()
    def _encode_pre(self, x):
        device_type = x.device.type
        if device_type == "cuda":
            with torch.autocast(device_type=device_type, dtype=self.autocast_dtype):
                z = self.model.encode_to_prequant(x)
        else:
            z = self.model.encode_to_prequant(x)
        return z

    @staticmethod
    def _first_tensor(out):
        """Pick the latent tensor out of the encode() return."""
        if isinstance(out, torch.Tensor):
            return out
        if isinstance(out, (tuple, list)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            return out[0]
        # diffusers-style posterior fallback
        if hasattr(out, "mode"):
            return out.mode()
        if hasattr(out, "sample"):
            return out.sample()
        raise TypeError(f"Could not extract latent tensor from encode() output of type {type(out).__name__}")

    @torch.no_grad()
    def extract(self, x, variant="default"):
        """
        Returns one (B, F) feature tensor. For non-VQ models `variant` is ignored.
        """
        if self.is_vq and variant == "pre":
            out = self._encode_pre(x)
        else:
            out = self._encode_raw(x)
        z = self._first_tensor(out)

        z = z.float()
        z = _normalize_to_BCdots(z)
        return _to_features(z, self.pooling)

    @torch.no_grad()
    def extract_pair(self, x):
        """
        For VQ: return {"pre": (B, F), "post": (B, F)} computed efficiently.
        For non-VQ: return {"default": (B, F)}.

        We can't trivially share the encoder forward across pre/post in this codebase
        (pre takes encoder->quant_conv only; post additionally runs the quantizer).
        Both internally call the encoder; that's a known cost. Still cheaper than
        running the full pipeline twice end-to-end.
        """
        if not self.is_vq:
            return {"default": self.extract(x)}
        return {
            "pre": self.extract(x, variant="pre"),
            "post": self.extract(x, variant="post"),
        }
