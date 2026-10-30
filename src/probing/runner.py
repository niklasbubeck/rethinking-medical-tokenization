import os
import csv
import json
import time
import numpy as np
import torch
from tqdm import tqdm

from .loader import load_first_stage
from .features import FeatureExtractor
from .cache import FeatureCache
from .datasets import build_probe_loaders, ALL_2D, get_dataset_spec
from .probe import train_linear_head


def _unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        return batch[0], batch[1]
    return batch["image"], batch["label"]


def _to_label_np(lbl):
    return lbl.numpy() if isinstance(lbl, torch.Tensor) else np.asarray(lbl)


def _extract_split_to_cache(split, loader, fx, variants, cache, dataset_name, device):
    """
    Stream features+labels for one split into per-variant fp16 memmaps.

    Memory cost is O(batch_size * feature_dim) instead of O(N * feature_dim * 2)
    that the old list+concatenate path paid (the *2 came from concatenate holding
    both inputs and outputs alive). On VQ models we extract both pre+post variants
    in the same loop without ever materialising the full tensor in RAM.
    """
    missing = [v for v in variants if not cache.has(v, dataset_name, split)]
    if not missing:
        return

    n_total = len(loader.dataset)

    # First batch: probe shapes (feature_dim, label shape/dtype) and write it.
    it = iter(loader)
    first = next(it)
    img0, lbl0 = _unpack_batch(first)
    img0 = img0.to(device, non_blocking=True)
    out0 = fx.extract_pair(img0) if len(missing) > 1 else {missing[0]: fx.extract(img0)}
    lbl0_np = _to_label_np(lbl0)

    feat_dim_by = {v: int(out0[v].shape[1]) for v in missing}
    label_shape = (n_total,) + lbl0_np.shape[1:]
    label_dtype = lbl0_np.dtype

    writers = {
        v: cache.open_writer(
            v, dataset_name, split,
            n=n_total, feature_dim=feat_dim_by[v],
            label_shape=label_shape, label_dtype=label_dtype,
        )
        for v in missing
    }
    try:
        for v in missing:
            writers[v].write(out0[v], lbl0_np)

        # Remaining batches.
        n_batches = max(0, len(loader) - 1)
        for batch in tqdm(it, total=n_batches, leave=False, dynamic_ncols=True):
            img, lbl = _unpack_batch(batch)
            img = img.to(device, non_blocking=True)
            out = fx.extract_pair(img) if len(missing) > 1 else {missing[0]: fx.extract(img)}
            lbl_np = _to_label_np(lbl)
            for v in missing:
                writers[v].write(out[v], lbl_np)

        for v in missing:
            writers[v].finalize()
    except BaseException:
        for w in writers.values():
            w.abort()
        raise


def _flatten_metrics_for_csv(test_metrics):
    """Flatten metrics dict; keep a stable column order for the CSV."""
    keys_order = ["top1", "auroc", "auroc_macro", "ap_macro", "qwk", "exact_match"]
    return {k: test_metrics.get(k, "") for k in keys_order}


def run_probing(cfg):
    """
    cfg keys (DictConfig or plain dict):
      models: list of model names
      datasets: list of dataset names (default: all 12 2D)
      base_dir, project, data_root, image_size
      pooling, normalize_features, batch_size, num_workers
      epochs, lr, weight_decay, head_batch_size
      run_name (for outputs), aggregate_dir
      device
      ckpt: optional int step to load
    """
    device = cfg.get("device", "cuda")
    image_size = int(cfg.get("image_size", 224))
    pooling = cfg.get("pooling", "mean")
    bs = int(cfg.get("batch_size", 256))
    num_workers = int(cfg.get("num_workers", 8))
    base_dir = cfg["base_dir"]
    project = cfg.get("project", "tokenizers")
    data_root = cfg["data_root"]
    run_name = cfg.get("run_name") or time.strftime("probe_%Y%m%d_%H%M%S")
    aggregate_dir = cfg.get("aggregate_dir") or os.path.join(base_dir, "probing", run_name)
    os.makedirs(aggregate_dir, exist_ok=True)

    models = list(cfg["models"])
    datasets = list(cfg.get("datasets") or ALL_2D)

    aggregate_rows = []  # list of dicts -> CSV rows
    aggregate_full = {"run_name": run_name, "config": dict(cfg), "results": {}}

    csv_path = os.path.join(aggregate_dir, "results.csv")
    json_path = os.path.join(aggregate_dir, "results.json")

    print(f"[probing] run_name={run_name}")
    print(f"[probing] models ({len(models)}): {models}")
    print(f"[probing] datasets ({len(datasets)}): {datasets}")
    print(f"[probing] aggregate dir: {aggregate_dir}")

    for model_name in models:
        print(f"\n========== MODEL: {model_name} ==========")
        info = load_first_stage(
            model_name, base_dir=base_dir, project=project, ckpt=cfg.get("ckpt"),
            img_size=image_size, device=device,
        )
        fx = FeatureExtractor(info["model"], family=info["family"], pooling=pooling)
        variants = fx.variants

        per_model_dir = os.path.join(info["model_dir"], "probing", run_name)
        os.makedirs(per_model_dir, exist_ok=True)
        cache = FeatureCache(root=os.path.join(info["model_dir"], "probing"), run_name=run_name)
        cache.set_meta(
            model=model_name, family=info["family"], ckpt_step=info["ckpt_step"],
            ckpt_path=info["ckpt_path"], image_size=image_size, pooling=pooling,
            variants=variants,
        )

        per_model_results = {
            "model": model_name, "family": info["family"], "ckpt_step": info["ckpt_step"],
            "image_size": image_size, "pooling": pooling, "variants": variants,
            "datasets": {},
        }

        for ds_name in datasets:
            spec = get_dataset_spec(ds_name)
            print(f"\n--- {model_name} :: {ds_name} (task={spec.task_type}, n_classes={spec.n_classes}) ---")
            loaders, _ = build_probe_loaders(
                ds_name, root=data_root, batch_size=bs, num_workers=num_workers,
                image_size=image_size, download=False,
            )
            t0 = time.time()
            for split, loader in loaders.items():
                _extract_split_to_cache(split, loader, fx, variants, cache, ds_name, device)
            print(f"[features] extract+cache time: {time.time() - t0:.1f}s")

            ds_results = {}
            for variant in variants:
                fx_tr, y_tr = cache.load(variant, ds_name, "train")
                fx_va, y_va = cache.load(variant, ds_name, "val")
                fx_te, y_te = cache.load(variant, ds_name, "test")

                res = train_linear_head(
                    fx_tr, y_tr, fx_va, y_va, fx_te, y_te,
                    spec=spec,
                    epochs=int(cfg.get("epochs", 50)),
                    batch_size=int(cfg.get("head_batch_size", 1024)),
                    lr=float(cfg.get("lr", 1e-3)),
                    weight_decay=float(cfg.get("weight_decay", 0.0)),
                    normalize=bool(cfg.get("normalize_features", True)),
                    device=device,
                    seed=int(cfg.get("seed", 0)),
                    verbose=bool(cfg.get("verbose", True)),
                )

                # Save head + per-(model,ds,variant) info
                heads_dir = os.path.join(per_model_dir, "heads")
                os.makedirs(heads_dir, exist_ok=True)
                head_path = os.path.join(heads_dir, f"{ds_name}.{variant}.pt")
                if res["head_state"] is not None:
                    torch.save({"state_dict": res["head_state"], "spec": spec.__dict__}, head_path)

                # Strip non-JSON-serializable bits before storing
                res_for_dump = {k: v for k, v in res.items() if k != "head_state"}
                ds_results[variant] = res_for_dump

                # Aggregate row
                row = {
                    "model": model_name,
                    "variant": variant,
                    "model:variant": f"{model_name}:{variant}" if variant != "default" else model_name,
                    "dataset": ds_name,
                    "task_type": spec.task_type,
                    "n_classes": spec.n_classes,
                    "feature_dim": res["feature_dim"],
                    "n_train": res["n_train"],
                    "n_val": res["n_val"],
                    "n_test": res["n_test"],
                    "primary_metric": spec.primary_metric,
                    "best_val_primary": res["best_val_metric"],
                    "elapsed_sec": round(res["elapsed_sec"], 2),
                    **{f"test_{k}": v for k, v in _flatten_metrics_for_csv(res["test_metrics"]).items()},
                }
                aggregate_rows.append(row)
                print(f"[done] {model_name}:{variant} :: {ds_name} -> "
                      f"val {spec.primary_metric}={res['best_val_metric']:.4f}, "
                      f"test {res['test_metrics']}")

            per_model_results["datasets"][ds_name] = ds_results

        # Per-model JSON
        with open(os.path.join(per_model_dir, "results.json"), "w") as f:
            json.dump(per_model_results, f, indent=2)
        aggregate_full["results"][model_name] = per_model_results

        # Free GPU memory before next model
        del fx, info
        torch.cuda.empty_cache()

        # Persist aggregate after each model so an interrupted run still leaves a usable table
        _write_aggregate(csv_path, json_path, aggregate_rows, aggregate_full)

    print(f"\n[probing] DONE. Aggregate: {csv_path}")
    return aggregate_full


def _write_aggregate(csv_path, json_path, rows, full):
    if rows:
        fieldnames = list(rows[0].keys())
        # extend with any keys present in later rows
        seen = set(fieldnames)
        for r in rows[1:]:
            for k in r:
                if k not in seen:
                    fieldnames.append(k); seen.add(k)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
    with open(json_path, "w") as f:
        json.dump(full, f, indent=2, default=str)
