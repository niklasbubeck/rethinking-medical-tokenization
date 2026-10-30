import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
)


def _to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _safe_auroc(y_true, y_score, **kw):
    try:
        return float(roc_auc_score(y_true, y_score, **kw))
    except ValueError:
        return float("nan")


def _safe_ap(y_true, y_score, **kw):
    try:
        return float(average_precision_score(y_true, y_score, **kw))
    except ValueError:
        return float("nan")


def compute_classification_metrics(logits, labels, task_type, n_classes):
    """
    logits: (N, C) torch tensor or ndarray
    labels: (N,) for multiclass/ordinal/binary, (N, C) for multilabel
    task_type: "multiclass" | "multilabel" | "ordinal"

    Returns: dict of float metrics. Keys depend on task_type.
    """
    logits = _to_np(logits).astype(np.float32)
    labels = _to_np(labels)

    out = {}

    if task_type == "multilabel":
        # sigmoid -> per-class probs
        probs = 1.0 / (1.0 + np.exp(-logits))
        if labels.ndim == 1:
            raise ValueError(f"multilabel expects (N, C) labels, got shape {labels.shape}")
        labels = labels.astype(np.int32)
        out["auroc_macro"] = _safe_auroc(labels, probs, average="macro")
        out["ap_macro"]    = _safe_ap(labels, probs, average="macro")
        # Fraction of fully-correct rows at threshold 0.5 (loose top-line, less informative than AP)
        preds = (probs >= 0.5).astype(np.int32)
        out["exact_match"] = float((preds == labels).all(axis=1).mean())
        return out

    # multiclass / ordinal / binary all reduce to softmax + argmax
    # Stable softmax
    z = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(z); probs /= probs.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)

    if labels.ndim > 1:
        labels = labels.reshape(-1)
    labels = labels.astype(np.int64)

    out["top1"] = float(accuracy_score(labels, preds))

    # AUROC: binary uses positive-class prob; multiclass uses macro one-vs-rest
    if n_classes == 2:
        out["auroc"] = _safe_auroc(labels, probs[:, 1])
        out["auroc_macro"] = out["auroc"]
    else:
        out["auroc_macro"] = _safe_auroc(labels, probs, multi_class="ovr", average="macro")

    if task_type == "ordinal":
        out["qwk"] = float(cohen_kappa_score(labels, preds, weights="quadratic"))

    return out


def primary_value(metrics, primary):
    """Pull the primary metric from a metrics dict, NaN-safe."""
    v = metrics.get(primary, float("nan"))
    return v if v == v else float("-inf")  # NaN -> -inf so it never wins early-stop comparisons
