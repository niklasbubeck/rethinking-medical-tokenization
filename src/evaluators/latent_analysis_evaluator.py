from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ..metrics.metrics import (
    GenerationMetric,
    LPIPSMetric,
    MAEMetric,
    MSEMetric,
    PSNRMetric,
    SSIMMetric,
)
from .base_evaluator import EvaluationTask

# Metrics that support (gt, pred) image tensors in latent-robustness tasks.
_IMAGE_METRIC_TYPES = (LPIPSMetric, MSEMetric, MAEMetric, SSIMMetric, PSNRMetric)


def _tensor_metric_to_float(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().mean())
    return float(value)


def _wrap_callable_metric(fn: Callable[..., torch.Tensor | float | int]) -> Callable[[torch.Tensor, torch.Tensor], float]:
    def wrapped(gt: torch.Tensor, pred: torch.Tensor) -> float:
        return _tensor_metric_to_float(fn(gt, pred))

    return wrapped


def _metric_obj_to_fn(task: object) -> Callable[[torch.Tensor, torch.Tensor], float]:
    def wrapped(gt: torch.Tensor, pred: torch.Tensor) -> float:
        return _tensor_metric_to_float(task(gt, pred))  # type: ignore[operator]

    return wrapped


def resolve_latent_perturbation_metric_fns(
    metric_fns: dict[str, Callable[..., object]] | list | tuple | None,
    metrics: list | tuple | None,
) -> dict[str, Callable[[torch.Tensor, torch.Tensor], float]]:
    """
    Build name -> (gt, pred) -> scalar for Hydra metric lists (see configs/tokenizer.yaml),
    explicit dicts of callables, or None (use `metrics`).
    Skips GenerationMetric (needs saved image paths, not in-memory tensors).
    """
    if metric_fns is not None and isinstance(metric_fns, dict):
        return {name: _wrap_callable_metric(fn) for name, fn in metric_fns.items()}

    tasks: list | tuple = ()
    if metric_fns is not None:
        tasks = metric_fns
    elif metrics is not None:
        tasks = metrics

    out: dict[str, Callable[[torch.Tensor, torch.Tensor], float]] = {}
    skipped_gen = False
    for task in tasks:
        if isinstance(task, GenerationMetric):
            skipped_gen = True
            continue
        if isinstance(task, _IMAGE_METRIC_TYPES):
            name = task.__class__.__name__
            out[name] = _metric_obj_to_fn(task)
    if skipped_gen:
        warnings.warn(
            "Latent perturbation robustness skips GenerationMetric (FID/IS need saved sample directories). "
            "Use pixel metrics (MSE, MAE, SSIM, PSNR, LPIPS) only.",
            UserWarning,
            stacklevel=3,
        )
    if not out:
        raise ValueError(
            "No usable image metrics for latent perturbation robustness. "
            "Pass a list of MSE/MAE/SSIM/PSNR/LPIPS metrics (not GenerationMetric), "
            "or a dict[str, callable(gt, pred)]."
        )
    return out


@dataclass
class EmbeddingResult:
    tsne: np.ndarray | None
    umap: np.ndarray | None


@dataclass
class EffectiveDimensionResult:
    effective_dim_90: int
    effective_dim_95: int
    effective_dim_99: int
    participation_ratio: float
    cumulative_variance: np.ndarray
    explained_variance: np.ndarray


class LatentAnalysisTaskBase(EvaluationTask):
    """
    Shared utilities for latent-space analysis tasks.
    """

    def __init__(
        self,
        name: str,
        metrics,
        latent_extractor: Callable[[torch.nn.Module, dict], torch.Tensor | np.ndarray] | None = None,
        input_key: str = "image",
        label_key: str = "label",
        dataset_id_key: str = "dataset_id",
        n_samples: int | None = None,
        save_as: list[str] | None = None,
        random_state: int = 42,
    ):
        super().__init__(name=name, metrics=metrics, input_key=input_key, save_as=save_as or ["npy"])
        self.latent_extractor = latent_extractor or self._default_latent_extractor
        self.input_key = input_key
        self.label_key = label_key
        self.dataset_id_key = dataset_id_key
        self.n_samples = n_samples
        self.random_state = random_state
        self.last_results: dict = {}

    def _default_latent_extractor(self, model, batch):
        """
        Prefer GenWrapper API, then fall back to generic encode outputs.
        """
        if hasattr(model, "vae_encode"):
            return model.vae_encode(batch[self.input_key])
        if hasattr(model, "encode"):
            encoded = model.encode(batch[self.input_key])
            if isinstance(encoded, (list, tuple)) and len(encoded) > 0:
                return encoded[0]
            return encoded
        raise AttributeError(
            "Model does not expose `vae_encode` or `encode`. "
            "Pass a custom `latent_extractor` to this task."
        )

    @staticmethod
    def _extract_tensor_like(latents):
        if isinstance(latents, (torch.Tensor, np.ndarray)):
            return latents
        if isinstance(latents, dict):
            for key in ("latents", "latent", "z", "quant", "embedding", "embeddings"):
                if key in latents:
                    return latents[key]
            if len(latents) > 0:
                return next(iter(latents.values()))
        if isinstance(latents, (list, tuple)):
            for item in latents:
                if isinstance(item, (torch.Tensor, np.ndarray)):
                    return item
            if len(latents) > 0:
                return latents[0]
        return latents

    @staticmethod
    def flatten_latents(latents) -> np.ndarray:
        latents = LatentAnalysisTaskBase._extract_tensor_like(latents)
        if isinstance(latents, torch.Tensor):
            latents = latents.detach().cpu().numpy()
        else:
            latents = np.asarray(latents)
        if latents.ndim < 2:
            raise ValueError(f"Expected latent tensor with >=2 dims, got {latents.shape}.")
        return latents.reshape(latents.shape[0], -1)

    def process_batch(self, model, batch):
        # Analysis tasks override evaluate() directly.
        raise NotImplementedError

    @torch.no_grad()
    def _collect_latents_and_labels(
        self,
        model,
        dataloader,
        require_labels: bool = False,
        require_dataset_ids: bool = False,
    ):
        model = self.unwrap_model(model)
        all_latents: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        all_dataset_ids: list[np.ndarray] = []
        collected = 0

        for batch in self.multi_meter.log_every(dataloader, len(dataloader), disable=not self.is_main):
            latents = self.latent_extractor(model, batch)
            latents = self.flatten_latents(latents)
            if self.n_samples is not None:
                remaining = self.n_samples - collected
                if remaining <= 0:
                    break
                latents = latents[:remaining]

            all_latents.append(latents)
            collected += len(latents)

            if require_labels:
                labels = batch[self.label_key]
                if isinstance(labels, torch.Tensor):
                    labels = labels.detach().cpu().numpy()
                labels = np.asarray(labels).reshape(-1)[: len(latents)]
                all_labels.append(labels)

            if require_dataset_ids:
                if self.dataset_id_key not in batch:
                    raise KeyError(
                        f"Batch does not contain `{self.dataset_id_key}` required for dataset-aware coloring."
                    )
                dataset_ids = batch[self.dataset_id_key]
                if isinstance(dataset_ids, torch.Tensor):
                    dataset_ids = dataset_ids.detach().cpu().numpy()
                dataset_ids = np.asarray(dataset_ids).reshape(-1)[: len(latents)]
                all_dataset_ids.append(dataset_ids)

            if self.n_samples is not None and collected >= self.n_samples:
                break

        if len(all_latents) == 0:
            raise RuntimeError("No latent samples were collected from the dataloader.")

        x = np.concatenate(all_latents, axis=0)
        y = np.concatenate(all_labels, axis=0) if require_labels else None
        d = np.concatenate(all_dataset_ids, axis=0) if require_dataset_ids else None
        return x, y, d

    def _task_output_dir(self):
        if self.output_dir is None or not self.is_main:
            return None
        task_dir = os.path.join(self.output_dir, self.name)
        os.makedirs(task_dir, exist_ok=True)
        return task_dir

    def _save_array(self, filename: str, data: np.ndarray):
        task_dir = self._task_output_dir()
        if task_dir is None:
            return
        np.save(os.path.join(task_dir, filename), data)

    def _save_scatter_figure(
        self,
        points_2d: np.ndarray,
        labels: np.ndarray | None,
        dataset_ids: np.ndarray | None,
        title: str,
        filename: str,
        point_size: float = 8.0,
        alpha: float = 0.7,
    ):
        task_dir = self._task_output_dir()
        if task_dir is None:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        if labels is None and dataset_ids is None:
            ax.scatter(points_2d[:, 0], points_2d[:, 1], s=point_size, alpha=alpha)
        elif labels is not None and dataset_ids is not None:
            # Hierarchical coloring:
            # - dataset_id uses distinct base colors
            # - class label uses subtones within each dataset.
            y = np.asarray(labels).reshape(-1).astype(str)
            d = np.asarray(dataset_ids).reshape(-1).astype(str)
            unique_ds = np.unique(d)
            base_palette = plt.get_cmap("tab10")
            ds_to_base = {
                ds: np.array(base_palette(i % base_palette.N)[:3], dtype=float)
                for i, ds in enumerate(unique_ds)
            }

            for ds in unique_ds:
                ds_mask = d == ds
                ds_classes = np.unique(y[ds_mask])
                cdenom = max(1, len(ds_classes) - 1)
                class_to_lighten = {cls: 0.10 + 0.70 * (j / cdenom) for j, cls in enumerate(ds_classes)}
                base_rgb = np.array(ds_to_base[ds], dtype=float)
                for cls in ds_classes:
                    m = ds_mask & (y == cls)
                    lighten = class_to_lighten[cls]
                    # Blend with white to create subtones of the dataset base color.
                    color_rgb = base_rgb * (1.0 - lighten) + np.ones(3) * lighten
                    ax.scatter(
                        points_2d[m, 0],
                        points_2d[m, 1],
                        color=[color_rgb],
                        s=point_size,
                        alpha=alpha,
                    )

            # Show dataset-level legend only (classes are encoded as subtones).
            from matplotlib.lines import Line2D

            handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=ds_to_base[ds],
                    markeredgecolor="none",
                    markersize=6,
                    label=f"dataset_id={ds}",
                )
                for ds in unique_ds
            ]
            ax.legend(handles=handles, title="Dataset IDs (base colors)", loc="best")
        else:
            y = np.asarray(labels).reshape(-1)
            # Convert arbitrary label types to integer ids for coloring.
            _, y_ids = np.unique(y.astype(str), return_inverse=True)
            scatter = ax.scatter(points_2d[:, 0], points_2d[:, 1], c=y_ids, cmap="tab20", s=point_size, alpha=alpha)
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label("label_id")
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(os.path.join(task_dir, filename), dpi=200)
        plt.close(fig)


class TSNEUMAPLatentAnalysis(LatentAnalysisTaskBase):
    """
    Task 1: t-SNE / UMAP embeddings of flattened latents.
    """

    def __init__(
        self,
        name: str,
        metrics,
        latent_extractor: Callable[[torch.nn.Module, dict], torch.Tensor | np.ndarray] | None = None,
        input_key: str = "image",
        label_key: str = "label",
        dataset_id_key: str = "dataset_id",
        n_samples: int | None = None,
        run_tsne: bool = True,
        run_umap: bool = True,
        tsne_perplexity: float = 30.0,
        tsne_n_iter: int = 1000,
        umap_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        export_figures: bool = True,
        point_size: float = 8.0,
        alpha: float = 0.7,
        random_state: int = 42,
    ):
        super().__init__(
            name=name,
            metrics=metrics,
            latent_extractor=latent_extractor,
            input_key=input_key,
            label_key=label_key,
            dataset_id_key=dataset_id_key,
            n_samples=n_samples,
            random_state=random_state,
        )
        self.run_tsne_flag = run_tsne
        self.run_umap_flag = run_umap
        self.tsne_perplexity = tsne_perplexity
        self.tsne_n_iter = tsne_n_iter
        self.umap_neighbors = umap_neighbors
        self.umap_min_dist = umap_min_dist
        self.export_figures = export_figures
        self.point_size = point_size
        self.alpha = alpha

    @torch.no_grad()
    def evaluate(self, model, dataloader, mode):
        x, y, d = self._collect_latents_and_labels(
            model,
            dataloader,
            require_labels=True,
            require_dataset_ids=True,
        )
        n = len(x)

        result = EmbeddingResult(tsne=None, umap=None)

        if self.run_tsne_flag:
            if n < 5:
                raise ValueError("t-SNE requires at least 5 samples.")
            safe_perplexity = max(2.0, min(float(self.tsne_perplexity), (n // 4) - 1))
            # sklearn API changed from n_iter -> max_iter in newer versions.
            try:
                tsne = TSNE(
                    n_components=2,
                    perplexity=safe_perplexity,
                    random_state=self.random_state,
                    max_iter=self.tsne_n_iter,
                )
            except TypeError:
                tsne = TSNE(
                    n_components=2,
                    perplexity=safe_perplexity,
                    random_state=self.random_state,
                    n_iter=self.tsne_n_iter,
                )
            result.tsne = tsne.fit_transform(x)
            self._save_array("tsne.npy", result.tsne)
            if self.export_figures:
                self._save_scatter_figure(
                    points_2d=result.tsne,
                    labels=y,
                    dataset_ids=d,
                    title=f"t-SNE ({self.name})",
                    filename="tsne.png",
                    point_size=self.point_size,
                    alpha=self.alpha,
                )

        if self.run_umap_flag:
            try:
                import umap  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "UMAP analysis requested but umap-learn is not installed. "
                    "Install with `pip install umap-learn`."
                ) from exc
            if n < 5:
                raise ValueError("UMAP requires at least 5 samples.")
            safe_neighbors = max(2, min(int(self.umap_neighbors), (n // 4) - 1))
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=safe_neighbors,
                min_dist=self.umap_min_dist,
                random_state=self.random_state,
            )
            result.umap = reducer.fit_transform(x)
            self._save_array("umap.npy", result.umap)
            if self.export_figures:
                self._save_scatter_figure(
                    points_2d=result.umap,
                    labels=y,
                    dataset_ids=d,
                    title=f"UMAP ({self.name})",
                    filename="umap.png",
                    point_size=self.point_size,
                    alpha=self.alpha,
                )

        self._save_array("labels.npy", y)
        if d is not None:
            self._save_array("dataset_ids.npy", d)
        self.last_results = {"tsne": result.tsne, "umap": result.umap, "labels": y, "dataset_ids": d}

        mean_metrics = {
            "n_samples": float(n),
            "tsne_computed": float(result.tsne is not None),
            "umap_computed": float(result.umap is not None),
        }
        return None, mean_metrics


class KNNRetrievalLatentAnalysis(LatentAnalysisTaskBase):
    """
    Task 3: k-NN retrieval in latent space (Recall@k).
    """

    def __init__(
        self,
        name: str,
        metrics,
        latent_extractor: Callable[[torch.nn.Module, dict], torch.Tensor | np.ndarray] | None = None,
        input_key: str = "image",
        label_key: str = "dataset_id",
        n_samples: int | None = None,
        ks: Iterable[int] = (1, 5, 10),
        metric: str = "euclidean",
        standardize: bool = True,
        random_state: int = 42,
    ):
        super().__init__(
            name=name,
            metrics=metrics,
            latent_extractor=latent_extractor,
            input_key=input_key,
            label_key=label_key,
            n_samples=n_samples,
            random_state=random_state,
        )
        self.ks = [int(k) for k in ks]
        self.metric = metric
        self.standardize = standardize

    @torch.no_grad()
    def evaluate(self, model, dataloader, mode):
        x, y, _ = self._collect_latents_and_labels(model, dataloader, require_labels=True)
        if self.standardize:
            x = StandardScaler().fit_transform(x)

        metrics_out: dict[str, float] = {"n_samples": float(len(x))}
        for k in self.ks:
            if len(x) <= k:
                raise ValueError(f"Number of samples ({len(x)}) must be > k ({k}).")
            nn = NearestNeighbors(n_neighbors=k + 1, metric=self.metric)
            nn.fit(x)
            indices = nn.kneighbors(x, return_distance=False)
            correct = 0
            for i in range(len(x)):
                neigh = indices[i, 1:]  # exclude query itself
                correct += int(y[i] in y[neigh])
            metrics_out[f"recall@{k}"] = float(correct / len(x))

        self.last_results = metrics_out
        return None, metrics_out


class PCAEffectiveDimensionLatentAnalysis(LatentAnalysisTaskBase):
    """
    Task 4: PCA-based effective / intrinsic latent dimension.
    """

    def __init__(
        self,
        name: str,
        metrics,
        latent_extractor: Callable[[torch.nn.Module, dict], torch.Tensor | np.ndarray] | None = None,
        input_key: str = "image",
        n_samples: int | None = None,
        random_state: int = 42,
        max_pca_samples: int = 2000_000,
        max_pca_features: int = 100_000,
        max_pca_components: int = 100_000,
        pca_svd_solver: str = "randomized",
        export_figures: bool = True,
        pca_plot_top_k: int = 50,
    ):
        super().__init__(
            name=name,
            metrics=metrics,
            latent_extractor=latent_extractor,
            input_key=input_key,
            n_samples=n_samples,
            random_state=random_state,
        )
        self.max_pca_samples = int(max_pca_samples)
        self.max_pca_features = int(max_pca_features)
        self.max_pca_components = int(max_pca_components)
        self.pca_svd_solver = pca_svd_solver
        self.export_figures = export_figures
        self.pca_plot_top_k = int(pca_plot_top_k)

    @staticmethod
    def _effective_dim_from_cumvar(cumvar: np.ndarray, threshold: float) -> int:
        if len(cumvar) == 0:
            return 0
        idx = int(np.searchsorted(cumvar, threshold, side="left"))
        idx = min(idx, len(cumvar) - 1)
        return idx + 1

    @staticmethod
    def participation_ratio(eigenvalues: np.ndarray) -> float:
        s1 = float(np.sum(eigenvalues))
        s2 = float(np.sum(eigenvalues**2))
        return (s1**2 / s2) if s2 > 0.0 else 0.0

    def _prepare_matrix_for_pca(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """
        Subsample rows/features to keep PCA tractable on very high-dimensional latents.
        """
        rng = np.random.default_rng(self.random_state)
        meta = {
            "n_samples_full": float(x.shape[0]),
            "n_features_full": float(x.shape[1]),
            "used_sample_subsampling": 0.0,
            "used_feature_subsampling": 0.0,
        }

        if x.shape[0] > self.max_pca_samples:
            sample_idx = rng.choice(x.shape[0], size=self.max_pca_samples, replace=False)
            x = x[sample_idx]
            meta["used_sample_subsampling"] = 1.0

        if x.shape[1] > self.max_pca_features:
            feature_idx = rng.choice(x.shape[1], size=self.max_pca_features, replace=False)
            x = x[:, feature_idx]
            meta["used_feature_subsampling"] = 1.0

        # Keep memory and SVD stable.
        x = x.astype(np.float32, copy=False)
        meta["n_samples_pca"] = float(x.shape[0])
        meta["n_features_pca"] = float(x.shape[1])
        return x, meta

    @torch.no_grad()
    def evaluate(self, model, dataloader, mode):
        x, _, _ = self._collect_latents_and_labels(model, dataloader, require_labels=False)
        x, pca_meta = self._prepare_matrix_for_pca(x)

        n_components = min(x.shape[0], x.shape[1], self.max_pca_components)
        pca = PCA(
            n_components=n_components,
            svd_solver=self.pca_svd_solver,
            random_state=self.random_state,
        )
        pca.fit(x)

        explained = pca.explained_variance_ratio_
        cumvar = np.cumsum(explained)
        result = EffectiveDimensionResult(
            effective_dim_90=self._effective_dim_from_cumvar(cumvar, 0.90),
            effective_dim_95=self._effective_dim_from_cumvar(cumvar, 0.95),
            effective_dim_99=self._effective_dim_from_cumvar(cumvar, 0.99),
            participation_ratio=self.participation_ratio(pca.explained_variance_),
            cumulative_variance=cumvar,
            explained_variance=explained,
        )

        self._save_array("pca_explained_variance.npy", result.explained_variance)
        self._save_array("pca_cumulative_variance.npy", result.cumulative_variance)
        if self.export_figures:
            self._save_pca_figures(result.explained_variance, result.cumulative_variance)
        self.last_results = {
            "effective_dim_90": result.effective_dim_90,
            "effective_dim_95": result.effective_dim_95,
            "effective_dim_99": result.effective_dim_99,
            "participation_ratio": result.participation_ratio,
            **pca_meta,
        }

        mean_metrics = {
            "n_samples": pca_meta["n_samples_full"],
            "n_features": pca_meta["n_features_full"],
            "n_samples_pca": pca_meta["n_samples_pca"],
            "n_features_pca": pca_meta["n_features_pca"],
            "used_sample_subsampling": pca_meta["used_sample_subsampling"],
            "used_feature_subsampling": pca_meta["used_feature_subsampling"],
            "effective_dim_90": float(result.effective_dim_90),
            "effective_dim_95": float(result.effective_dim_95),
            "effective_dim_99": float(result.effective_dim_99),
            "participation_ratio": float(result.participation_ratio),
            "cumvar_last": float(cumvar[-1]) if len(cumvar) > 0 else 0.0,
            "var_top1": float(explained[0]) if len(explained) > 0 else 0.0,
            "var_top5": float(np.sum(explained[:5])) if len(explained) >= 5 else float(np.sum(explained)),
            "var_top10": float(np.sum(explained[:10])) if len(explained) >= 10 else float(np.sum(explained)),
        }
        return None, mean_metrics

    def _save_pca_figures(self, explained: np.ndarray, cumvar: np.ndarray):
        task_dir = self._task_output_dir()
        if task_dir is None:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        # Cumulative explained variance curve.
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(1, len(cumvar) + 1)
        ax.plot(x, cumvar, linewidth=2)
        for t in (0.90, 0.95, 0.99):
            ax.axhline(t, linestyle="--", linewidth=1)
        ax.set_xlabel("Number of components")
        ax.set_ylabel("Cumulative explained variance")
        ax.set_title(f"PCA cumulative variance ({self.name})")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(task_dir, "pca_cumulative_variance.png"), dpi=200)
        plt.close(fig)

        # Top-k explained variance per component.
        top_k = min(self.pca_plot_top_k, len(explained))
        if top_k > 0:
            fig, ax = plt.subplots(figsize=(8, 5))
            xx = np.arange(1, top_k + 1)
            ax.bar(xx, explained[:top_k])
            ax.set_xlabel("Principal component")
            ax.set_ylabel("Explained variance ratio")
            ax.set_title(f"PCA top-{top_k} explained variance ({self.name})")
            fig.tight_layout()
            fig.savefig(os.path.join(task_dir, "pca_top_explained_variance.png"), dpi=200)
            plt.close(fig)


class ContinuousLatentPerturbationRobustness(EvaluationTask):
    """
    Task 5 (continuous): perturb latent vectors with additive noise and measure
    reconstruction quality degradation.
    """

    def __init__(
        self,
        name: str,
        metrics,
        metric_fns: dict[str, Callable[..., object]] | list | tuple | None = None,
        encode_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor] | None = None,
        decode_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor] | None = None,
        input_key: str = "image",
        noise_stds: Iterable[float] = (0.0, 0.01, 0.05, 0.1),
        noise_type: str = "gaussian",
        n_samples: int | None = None,
        export_figures: bool = True,
        export_example_images: bool = True,
        n_example_images: int = 4,
        save_as: list[str] | None = None,
    ):
        super().__init__(name=name, metrics=metrics, input_key=input_key, save_as=save_as or ["npy"])
        self.encode_fn = encode_fn or self._default_encode_fn
        self.decode_fn = decode_fn or self._default_decode_fn
        self.metric_fns = resolve_latent_perturbation_metric_fns(metric_fns, metrics)
        self.input_key = input_key
        self.noise_stds = [float(x) for x in noise_stds]
        self.noise_type = noise_type
        self.n_samples = n_samples
        self.export_figures = export_figures
        self.export_example_images = export_example_images
        self.n_example_images = max(1, int(n_example_images))
        self.last_curve: list[dict[str, float]] = []

    def process_batch(self, model, batch):
        raise NotImplementedError

    @staticmethod
    def _default_encode_fn(model, images: torch.Tensor) -> torch.Tensor:
        return model.encode(images)

    @staticmethod
    def _default_decode_fn(model, latents: torch.Tensor) -> torch.Tensor:
        return model.decode(latents)

    @staticmethod
    def perturb_latent_continuous(z: torch.Tensor, noise_std: float, noise_type: str = "gaussian"):
        if noise_type == "gaussian":
            noise = torch.randn_like(z) * noise_std
        elif noise_type == "uniform":
            noise = (torch.rand_like(z) - 0.5) * (2.0 * noise_std)
        else:
            raise ValueError(f"Unsupported noise_type: {noise_type}")
        return z + noise

    @staticmethod
    def _tensor_chw_to_imshow_array(t: torch.Tensor) -> np.ndarray:
        """Single image [C, H, W] in [0, 1] -> HxW or HxWxC for matplotlib."""
        t = t.detach().float().cpu()
        if t.shape[0] == 1:
            return t[0].numpy()
        return np.transpose(t.numpy(), (1, 2, 0))

    def _save_example_gt_recon_panels(
        self,
        images: torch.Tensor,
        recon: torch.Tensor,
        noise_std: float,
        examples_dir: str,
    ) -> None:
        if self.output_dir is None or not self.is_main:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        k = min(self.n_example_images, images.shape[0], recon.shape[0])
        if k < 1:
            return
        img_disp, recon_disp = self._normalize_output(images[:k], recon[:k])
        fig, axes = plt.subplots(k, 2, figsize=(6, 2.2 * k), squeeze=False)
        for i in range(k):
            g = self._tensor_chw_to_imshow_array(img_disp[i])
            r = self._tensor_chw_to_imshow_array(recon_disp[i])
            cmap = "gray" if g.ndim == 2 else None
            axes[i, 0].imshow(g, cmap=cmap, vmin=0.0, vmax=1.0)
            axes[i, 0].set_title("GT")
            axes[i, 0].axis("off")
            axes[i, 1].imshow(r, cmap=cmap, vmin=0.0, vmax=1.0)
            axes[i, 1].set_title(f"recon (σ={noise_std:g})")
            axes[i, 1].axis("off")
        fig.suptitle(f"Latent noise σ={noise_std:g} ({self.name})")
        fig.tight_layout()
        tag = f"{noise_std:.6f}".rstrip("0").rstrip(".")
        if tag == "" or tag == "-":
            tag = str(noise_std)
        os.makedirs(examples_dir, exist_ok=True)
        fig.savefig(os.path.join(examples_dir, f"gt_vs_recon_noise_{tag}.png"), dpi=160)
        plt.close(fig)

    @torch.no_grad()
    def evaluate(self, model, dataloader, mode):
        model = self.unwrap_model(model)
        results: list[dict[str, float]] = []

        for noise_std in self.noise_stds:
            metric_buffer = {name: [] for name in self.metric_fns}
            n_seen = 0
            first_batch_saved = False
            for batch in self.multi_meter.log_every(dataloader, len(dataloader), disable=not self.is_main):
                images = batch[self.input_key]
                if self.n_samples is not None and n_seen >= self.n_samples:
                    break

                z, *_ = self.encode_fn(model, images)
                z_pert = self.perturb_latent_continuous(z, noise_std, self.noise_type)
                recon = self.decode_fn(model, z_pert)

                if (
                    self.export_figures
                    and self.export_example_images
                    and self.output_dir is not None
                    and self.is_main
                    and not first_batch_saved
                ):
                    task_dir = os.path.join(self.output_dir, self.name)
                    self._save_example_gt_recon_panels(
                        images,
                        recon,
                        noise_std,
                        os.path.join(task_dir, "examples"),
                    )
                    first_batch_saved = True

                for metric_name, metric_fn in self.metric_fns.items():
                    metric_buffer[metric_name].append(metric_fn(images, recon))
                n_seen += len(images)

            row = {"noise_std": noise_std}
            for metric_name, values in metric_buffer.items():
                row[metric_name] = float(np.mean(values)) if values else float("nan")
            results.append(row)

        self.last_curve = results
        if self.output_dir is not None and self.is_main:
            task_dir = os.path.join(self.output_dir, self.name)
            os.makedirs(task_dir, exist_ok=True)
            np.save(os.path.join(task_dir, "continuous_robustness_curve.npy"), np.array(results, dtype=object))
            if self.export_figures:
                self._save_robustness_plot(
                    results=results,
                    x_key="noise_std",
                    title=f"Continuous latent robustness ({self.name})",
                    filename="continuous_robustness_curve.png",
                )
        mean_metrics = {}
        if results:
            for metric_name in self.metric_fns.keys():
                mean_metrics[f"{metric_name}_at_noise_0"] = float(
                    next((r[metric_name] for r in results if r["noise_std"] == 0.0), results[0][metric_name])
                )
                mean_metrics[f"{metric_name}_at_noise_max"] = float(results[-1][metric_name])
        return None, mean_metrics

    def _save_robustness_plot(self, results: list[dict[str, float]], x_key: str, title: str, filename: str):
        if self.output_dir is None or not self.is_main:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        if not results:
            return
        task_dir = os.path.join(self.output_dir, self.name)
        os.makedirs(task_dir, exist_ok=True)
        x = [row[x_key] for row in results]
        metric_names = [k for k in results[0].keys() if k != x_key]
        fig, ax = plt.subplots(figsize=(8, 5))
        for metric_name in metric_names:
            y = [row.get(metric_name, np.nan) for row in results]
            ax.plot(x, y, marker="o", label=metric_name)
        ax.set_xlabel(x_key)
        ax.set_ylabel("metric value")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if len(metric_names) > 0:
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(task_dir, filename), dpi=200)
        plt.close(fig)


class DiscreteLatentPerturbationRobustness(EvaluationTask):
    """
    Task 5 (discrete): perturb token indices via random flips and measure
    reconstruction quality degradation.
    """

    def __init__(
        self,
        name: str,
        metrics,
        codebook_size: int,
        metric_fns: dict[str, Callable[..., object]] | list | tuple | None = None,
        encode_indices_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor] | None = None,
        decode_indices_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor] | None = None,
        input_key: str = "image",
        flip_probs: Iterable[float] = (0.0, 0.01, 0.05, 0.1),
        n_samples: int | None = None,
        export_figures: bool = True,
        save_as: list[str] | None = None,
    ):
        super().__init__(name=name, metrics=metrics, input_key=input_key, save_as=save_as or ["npy"])
        self.encode_indices_fn = encode_indices_fn or self._default_encode_indices_fn
        self.decode_indices_fn = decode_indices_fn or self._default_decode_indices_fn
        self.metric_fns = resolve_latent_perturbation_metric_fns(metric_fns, metrics)
        self.codebook_size = int(codebook_size)
        self.input_key = input_key
        self.flip_probs = [float(x) for x in flip_probs]
        self.n_samples = n_samples
        self.export_figures = export_figures
        self.last_curve: list[dict[str, float]] = []

    def process_batch(self, model, batch):
        raise NotImplementedError

    @staticmethod
    def _default_encode_indices_fn(model, images: torch.Tensor) -> torch.Tensor:
        if not hasattr(model, "vae_encode"):
            raise AttributeError(
                "Model does not expose `vae_encode`. Pass a custom `encode_indices_fn` "
                "for discrete perturbation analysis."
            )
        return model.vae_encode(images)

    @staticmethod
    def _default_decode_indices_fn(model, indices: torch.Tensor) -> torch.Tensor:
        if not hasattr(model, "vae_decode"):
            raise AttributeError(
                "Model does not expose `vae_decode`. Pass a custom `decode_indices_fn` "
                "for discrete perturbation analysis."
            )
        return model.vae_decode(indices)

    @staticmethod
    def perturb_latent_discrete(indices: torch.Tensor, codebook_size: int, flip_prob: float):
        if codebook_size <= 1:
            raise ValueError("codebook_size must be > 1.")
        if not (0.0 <= flip_prob <= 1.0):
            raise ValueError("flip_prob must be in [0, 1].")

        mask = torch.rand_like(indices.float()) < flip_prob
        random_codes = torch.randint(0, codebook_size, indices.shape, device=indices.device)
        same = random_codes == indices
        random_codes[same] = (random_codes[same] + 1) % codebook_size
        return torch.where(mask, random_codes, indices)

    @torch.no_grad()
    def evaluate(self, model, dataloader, mode):
        model = self.unwrap_model(model)
        results: list[dict[str, float]] = []

        for flip_prob in self.flip_probs:
            metric_buffer = {name: [] for name in self.metric_fns}
            n_seen = 0
            for batch in self.multi_meter.log_every(dataloader, len(dataloader), disable=not self.is_main):
                images = batch[self.input_key]
                if self.n_samples is not None and n_seen >= self.n_samples:
                    break

                indices = self.encode_indices_fn(model, images)
                if not torch.is_tensor(indices) or indices.dtype not in (
                    torch.int8,
                    torch.int16,
                    torch.int32,
                    torch.int64,
                    torch.uint8,
                ):
                    raise ValueError(
                        "DiscreteLatentPerturbationRobustness expects integer token indices from "
                        "`encode_indices_fn`. Your model returned non-integer latents; use "
                        "ContinuousLatentPerturbationRobustness or pass custom encode/decode fns."
                    )
                perturbed = self.perturb_latent_discrete(indices, self.codebook_size, flip_prob)
                recon = self.decode_indices_fn(model, perturbed)

                for metric_name, metric_fn in self.metric_fns.items():
                    metric_buffer[metric_name].append(metric_fn(images, recon))
                n_seen += len(images)

            row = {"flip_prob": flip_prob}
            for metric_name, values in metric_buffer.items():
                row[metric_name] = float(np.mean(values)) if values else float("nan")
            results.append(row)

        self.last_curve = results
        if self.output_dir is not None and self.is_main:
            task_dir = os.path.join(self.output_dir, self.name)
            os.makedirs(task_dir, exist_ok=True)
            np.save(os.path.join(task_dir, "discrete_robustness_curve.npy"), np.array(results, dtype=object))
            if self.export_figures:
                self._save_robustness_plot(
                    results=results,
                    x_key="flip_prob",
                    title=f"Discrete latent robustness ({self.name})",
                    filename="discrete_robustness_curve.png",
                )
        mean_metrics = {}
        if results:
            for metric_name in self.metric_fns.keys():
                mean_metrics[f"{metric_name}_at_flip_0"] = float(
                    next((r[metric_name] for r in results if r["flip_prob"] == 0.0), results[0][metric_name])
                )
                mean_metrics[f"{metric_name}_at_flip_max"] = float(results[-1][metric_name])
        return None, mean_metrics

    def _save_robustness_plot(self, results: list[dict[str, float]], x_key: str, title: str, filename: str):
        if self.output_dir is None or not self.is_main:
            return
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        if not results:
            return
        task_dir = os.path.join(self.output_dir, self.name)
        os.makedirs(task_dir, exist_ok=True)
        x = [row[x_key] for row in results]
        metric_names = [k for k in results[0].keys() if k != x_key]
        fig, ax = plt.subplots(figsize=(8, 5))
        for metric_name in metric_names:
            y = [row.get(metric_name, np.nan) for row in results]
            ax.plot(x, y, marker="o", label=metric_name)
        ax.set_xlabel(x_key)
        ax.set_ylabel("metric value")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        if len(metric_names) > 0:
            ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(task_dir, filename), dpi=200)
        plt.close(fig)
