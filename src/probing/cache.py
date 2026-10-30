import os
import json
import numpy as np
import torch


class _StreamingWriter:
    """
    Fixed-size, per-(variant, dataset, split) writer that streams batches into an
    on-disk fp16 memmap. Atomic: writes go to <path>.tmp, then rename on finalize().
    """

    def __init__(self, cache, variant, dataset, split, feat_path, lbl_path,
                 n, feature_dim, label_shape, label_dtype):
        self.cache = cache
        self.variant = variant
        self.dataset = dataset
        self.split = split
        self.feat_path = feat_path
        self.lbl_path = lbl_path
        self.feat_path_tmp = feat_path + ".tmp.npy"
        self.lbl_path_tmp = lbl_path + ".tmp.npy"
        self.n = int(n)
        self.feature_dim = int(feature_dim)

        self.feat_mm = np.lib.format.open_memmap(
            self.feat_path_tmp, mode="w+", dtype=np.float16,
            shape=(self.n, self.feature_dim),
        )
        self.labels = np.empty(label_shape, dtype=label_dtype)
        self.pos = 0

    def write(self, feats, labels):
        """
        feats: torch.Tensor (B, F) on any device, any dtype.
        labels: numpy array, shape (B, ...) matching label_shape[1:].
        """
        b = int(feats.shape[0])
        end = self.pos + b
        if end > self.n:
            raise RuntimeError(f"writer overflow: pos+b={end} > n={self.n}")
        # fp16 cast before crossing PCIe; minimizes both copies and host RAM peak.
        feats_np = feats.to(torch.float16).cpu().numpy()
        self.feat_mm[self.pos:end] = feats_np
        self.labels[self.pos:end] = labels
        self.pos = end

    def finalize(self):
        if self.pos != self.n:
            raise RuntimeError(
                f"writer underflow for {self.dataset}/{self.split}/{self.variant}: "
                f"wrote {self.pos} but expected {self.n}"
            )
        self.feat_mm.flush()
        # Drop mmap handle before rename so kernel releases it cleanly.
        del self.feat_mm
        np.save(self.lbl_path_tmp, self.labels)
        os.replace(self.feat_path_tmp, self.feat_path)
        os.replace(self.lbl_path_tmp, self.lbl_path)

        ds_entry = self.cache._manifest.setdefault("entries", {}).setdefault(self.dataset, {})
        ds_entry.setdefault(self.variant, {})[self.split] = {
            "features_path": os.path.relpath(self.feat_path, self.cache.run_dir),
            "labels_path": os.path.relpath(self.lbl_path, self.cache.run_dir),
            "n": self.n,
            "feature_dim": self.feature_dim,
            "labels_shape": list(self.labels.shape),
            "labels_dtype": str(self.labels.dtype),
        }
        self.cache._save_manifest()

    def abort(self):
        try:
            del self.feat_mm
        except Exception:
            pass
        for p in (self.feat_path_tmp, self.lbl_path_tmp):
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass


class FeatureCache:
    """
    Simple .npy-backed cache for extracted features and labels.

    Layout under <root>/<run_name>/:
      manifest.json
      features/
        <variant>/<dataset>_<split>.npy           (N, F) float16
        <variant>/<dataset>_<split>.labels.npy    (N,) int64 or (N, C) for multilabel

    The manifest records what the cache was built with so we can detect stale entries
    (different ckpt step, different image size, different pooling).
    """

    def __init__(self, root, run_name):
        self.root = root
        self.run_name = run_name
        self.run_dir = os.path.join(root, run_name)
        self.feat_dir = os.path.join(self.run_dir, "features")
        os.makedirs(self.feat_dir, exist_ok=True)
        self.manifest_path = os.path.join(self.run_dir, "manifest.json")
        self._manifest = self._load_manifest()

    def _load_manifest(self):
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path) as f:
                return json.load(f)
        return {}

    def _save_manifest(self):
        with open(self.manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2, sort_keys=True)

    # Keys that, if changed, invalidate the entire cache. Lock these in set_meta().
    _CACHE_KEY_FIELDS = ("model", "ckpt_step", "image_size", "pooling")

    def set_meta(self, **kwargs):
        """
        Record/update manifest metadata. If the run was previously written with a
        different value for any cache-key field (model, ckpt_step, image_size,
        pooling), raise loudly so we don't silently reuse stale features.
        Use a fresh `run_name` to switch any of those.
        """
        existing = self._manifest.get("meta", {})
        conflicts = {
            k: (existing[k], kwargs[k])
            for k in self._CACHE_KEY_FIELDS
            if k in existing and k in kwargs and existing[k] != kwargs[k]
        }
        if conflicts:
            raise RuntimeError(
                f"Cache at {self.run_dir} was built with different settings: "
                f"{conflicts}. Use a new --exp_name to start a fresh cache."
            )
        self._manifest.setdefault("meta", {}).update(kwargs)
        self._save_manifest()

    @property
    def meta(self):
        return self._manifest.get("meta", {})

    def _paths(self, variant, dataset, split):
        d = os.path.join(self.feat_dir, variant)
        os.makedirs(d, exist_ok=True)
        feats = os.path.join(d, f"{dataset}_{split}.npy")
        labels = os.path.join(d, f"{dataset}_{split}.labels.npy")
        return feats, labels

    def has(self, variant, dataset, split):
        f, l = self._paths(variant, dataset, split)
        return os.path.exists(f) and os.path.exists(l)

    def save(self, variant, dataset, split, features, labels):
        f, l = self._paths(variant, dataset, split)
        np.save(f, features.astype(np.float16, copy=False))
        np.save(l, labels)
        ds_entry = self._manifest.setdefault("entries", {}).setdefault(dataset, {})
        ds_entry.setdefault(variant, {})[split] = {
            "features_path": os.path.relpath(f, self.run_dir),
            "labels_path": os.path.relpath(l, self.run_dir),
            "n": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
            "labels_shape": list(labels.shape),
            "labels_dtype": str(labels.dtype),
        }
        self._save_manifest()

    def open_writer(self, variant, dataset, split, n, feature_dim, label_shape, label_dtype):
        """Return a _StreamingWriter that writes into <feat_path> via memmap (fp16, atomic)."""
        f, l = self._paths(variant, dataset, split)
        return _StreamingWriter(
            self, variant, dataset, split, f, l,
            n=n, feature_dim=feature_dim,
            label_shape=label_shape, label_dtype=label_dtype,
        )

    def load(self, variant, dataset, split):
        f, l = self._paths(variant, dataset, split)
        features = np.load(f)
        labels = np.load(l)
        return features, labels
