import time
import numpy as np
import torch
import torch.nn as nn

from .metrics import compute_classification_metrics, primary_value


def _compute_norm_stats(arr, chunk=4096):
    """
    Compute per-feature (mean, std) in fp32 from a possibly-fp16 ndarray, in chunks.
    Avoids materializing the full fp32 cast (which for big flatten features can be tens of GB).
    Returns mean and std as fp32 (1, F) arrays.
    """
    n, fdim = int(arr.shape[0]), int(arr.shape[1])
    s  = np.zeros(fdim, dtype=np.float64)
    s2 = np.zeros(fdim, dtype=np.float64)
    for i in range(0, n, chunk):
        c = arr[i:i + chunk].astype(np.float32, copy=False)
        s  += c.sum(axis=0, dtype=np.float64)
        s2 += (c.astype(np.float64) ** 2).sum(axis=0)
    mean = (s / n).astype(np.float32)
    var  = (s2 / n - (s / n) ** 2).astype(np.float32)
    std  = np.sqrt(np.maximum(var, 0.0)) + np.float32(1e-6)
    return mean.reshape(1, -1), std.reshape(1, -1)


def _make_loss(task_type):
    if task_type == "multilabel":
        return nn.BCEWithLogitsLoss()
    return nn.CrossEntropyLoss()


def _to_label_tensor(labels, task_type):
    """CPU label tensor in the right dtype/shape for the chosen loss."""
    labels = np.asarray(labels)
    if task_type == "multilabel":
        if labels.ndim == 1:
            raise ValueError(f"multilabel needs (N, C) labels, got {labels.shape}")
        return torch.from_numpy(labels.astype(np.float32))
    if labels.ndim > 1:
        labels = labels.reshape(-1)
    return torch.from_numpy(labels.astype(np.int64))


@torch.no_grad()
def evaluate_head(head, x_cpu, y_for_metric, task_type, n_classes, device,
                  batch_size=4096, mean_t=None, std_t=None):
    """Stream features CPU->GPU per minibatch, optionally normalize on-device, compute metrics."""
    head.eval()
    n = x_cpu.shape[0]
    logits_chunks = []
    for i in range(0, n, batch_size):
        xb = x_cpu[i:i + batch_size].to(device, non_blocking=True).float()
        if mean_t is not None:
            xb = (xb - mean_t) / std_t
        logits_chunks.append(head(xb).float().cpu())
    logits = torch.cat(logits_chunks, dim=0)
    metrics = compute_classification_metrics(logits, y_for_metric, task_type=task_type, n_classes=n_classes)
    return metrics, logits


def train_linear_head(
    feats_train, y_train,
    feats_val,   y_val,
    feats_test,  y_test,
    spec,                       # DatasetSpec
    epochs=50,
    batch_size=1024,
    lr=1e-3,
    weight_decay=0.0,
    normalize=True,
    device="cuda",
    seed=0,
    verbose=True,
):
    """
    Train a single nn.Linear head on cached features.

    Features stay on CPU as pinned float32 tensors; mini-batches are shuttled to GPU.
    This lets the probe scale to flattened latents (e.g. 100k dim) that don't fit
    on the GPU as a single tensor.
    """
    torch.manual_seed(seed); np.random.seed(seed)

    # Keep features in their loaded dtype (typically fp16). NO fp32 cast and NO
    # pin_memory -- both would double host RAM. For lfq.f4 + tissuemnist that
    # difference is ~10 GB vs ~80 GB. We cast to fp32 on-GPU per-batch instead.
    feats_train = np.ascontiguousarray(feats_train)
    feats_val   = np.ascontiguousarray(feats_val)
    feats_test  = np.ascontiguousarray(feats_test)
    if normalize:
        fmean, fstd = _compute_norm_stats(feats_train)
    else:
        fmean = fstd = None

    F_dim = feats_train.shape[1]
    n_out = spec.n_classes

    x_tr = torch.from_numpy(feats_train)
    x_va = torch.from_numpy(feats_val)
    x_te = torch.from_numpy(feats_test)
    y_tr = _to_label_tensor(y_train, spec.task_type)

    fmean_t = torch.from_numpy(fmean).to(device) if fmean is not None else None
    fstd_t  = torch.from_numpy(fstd ).to(device) if fstd  is not None else None
    y_va_for_metric = y_val if spec.task_type == "multilabel" else np.asarray(y_val).reshape(-1).astype(np.int64)
    y_te_for_metric = y_test if spec.task_type == "multilabel" else np.asarray(y_test).reshape(-1).astype(np.int64)
    y_tr_for_metric = y_train if spec.task_type == "multilabel" else np.asarray(y_train).reshape(-1).astype(np.int64)

    head = nn.Linear(F_dim, n_out).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    loss_fn = _make_loss(spec.task_type)

    n_train = x_tr.shape[0]
    best_val = float("-inf")
    best_state = None
    history = []
    t0 = time.time()

    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(n_train)  # CPU permutation, indexes the CPU tensors
        running = 0.0
        n_batches = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb = x_tr.index_select(0, idx).to(device, non_blocking=True).float()
            if fmean_t is not None:
                xb = (xb - fmean_t) / fstd_t
            yb = y_tr.index_select(0, idx).to(device, non_blocking=True)
            logits = head(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            running += float(loss.detach())
            n_batches += 1
        sched.step()

        val_metrics, _ = evaluate_head(head, x_va, y_va_for_metric, spec.task_type, n_out,
                                       device=device, mean_t=fmean_t, std_t=fstd_t)
        v = primary_value(val_metrics, spec.primary_metric)
        history.append({"epoch": epoch, "train_loss": running / max(1, n_batches), "val": val_metrics})

        if v > best_val:
            best_val = v
            best_state = {k: t.detach().cpu().clone() for k, t in head.state_dict().items()}

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            print(f"  [probe {spec.name}] ep {epoch:>3d}/{epochs} "
                  f"loss={running / max(1, n_batches):.4f} "
                  f"val_{spec.primary_metric}={val_metrics.get(spec.primary_metric, float('nan')):.4f} "
                  f"best={best_val:.4f}")

    if best_state is not None:
        head.load_state_dict(best_state)
    test_metrics, _  = evaluate_head(head, x_te, y_te_for_metric, spec.task_type, n_out,
                                     device=device, mean_t=fmean_t, std_t=fstd_t)
    train_metrics, _ = evaluate_head(head, x_tr, y_tr_for_metric, spec.task_type, n_out,
                                     device=device, mean_t=fmean_t, std_t=fstd_t)

    return {
        "head_state": best_state,
        "best_val_metric": best_val,
        "best_val_metric_name": spec.primary_metric,
        "val_metrics": history[-1]["val"] if history else {},
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "feature_dim": F_dim,
        "n_train": int(n_train),
        "n_val": int(x_va.shape[0]),
        "n_test": int(x_te.shape[0]),
        "elapsed_sec": time.time() - t0,
        # Skip storing huge per-feature mean/std arrays in JSON for high-dim cases.
        "norm_mean_dim": int(fmean.shape[1]) if fmean is not None else 0,
        "norm_std_dim":  int(fstd.shape[1])  if fstd  is not None else 0,
        "history": history,
    }
