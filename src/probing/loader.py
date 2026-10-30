import os
import re
import glob
import torch
from medlat import get_model, available_models


def _strip_module_prefix(state_dict):
    return {(k[len("module."):] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def find_model_dir(name, base_dir, project="tokenizers"):
    """
    Resolve <base_dir>/outputs/<project>/<name>/. Raise FileNotFoundError if missing.
    """
    candidate = os.path.join(base_dir, "outputs", project, name)
    if not os.path.isdir(candidate):
        raise FileNotFoundError(
            f"Model directory not found: {candidate}. "
            f"Pass --base_dir / --project to point at the right location."
        )
    return candidate


def find_latest_ckpt(model_dir):
    """
    Return (path, step) of the highest-numbered ckpt.<step>.pt under <model_dir>/train/models/.
    """
    pattern = os.path.join(model_dir, "train", "models", "ckpt.*.pt")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No checkpoints matching {pattern}")

    def step_of(p):
        m = re.search(r"ckpt\.(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1

    files.sort(key=step_of)
    best = files[-1]
    return best, step_of(best)


def load_first_stage(name, base_dir, project="tokenizers", ckpt=None, img_size=224, device="cuda"):
    """
    Build a first-stage encoder by name (via medlat.get_model) and load its trained weights.

    Args:
        name: model name in the medlat registry (e.g. "continuous.aekl.f4_d32").
        base_dir: root that contains outputs/<project>/<name>/.
        project: subdir inside outputs/ (default "tokenizers").
        ckpt: int step to load, or absolute path. None -> latest in the model dir.
        img_size: image size to build the model with (must match training).
        device: target device for the loaded model.

    Returns:
        dict with: model (eval, frozen), model_dir, ckpt_path, ckpt_step, family, name.
    """
    if name not in available_models():
        raise KeyError(
            f"'{name}' is not in medlat.available_models(). "
            f"Did you mix up the encoder name with a generator dir like 'dit_<name>'?"
        )

    model_dir = find_model_dir(name, base_dir=base_dir, project=project)

    if ckpt is None:
        ckpt_path, ckpt_step = find_latest_ckpt(model_dir)
    elif isinstance(ckpt, int):
        ckpt_path = os.path.join(model_dir, "train", "models", f"ckpt.{ckpt}.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Requested checkpoint not found: {ckpt_path}")
        ckpt_step = ckpt
    else:
        ckpt_path = str(ckpt)
        m = re.search(r"ckpt\.(\d+)\.pt$", ckpt_path)
        ckpt_step = int(m.group(1)) if m else -1

    print(f"[probing.loader] building model: {name} (img_size={img_size})")
    model = get_model(name, img_size=img_size)

    print(f"[probing.loader] loading ckpt: {ckpt_path} (step={ckpt_step})")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_sd = state["model"] if isinstance(state, dict) and "model" in state else state
    sd = _strip_module_prefix(raw_sd)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[probing.loader] WARN missing keys (showing 5): {missing[:5]} ... (+{max(0,len(missing)-5)} more)")
    if unexpected:
        print(f"[probing.loader] WARN unexpected keys (showing 5): {unexpected[:5]} ... (+{max(0,len(unexpected)-5)} more)")

    model.eval().requires_grad_(False)
    model.to(device)

    family = name.split(".", 1)[0]  # continuous | discrete | token | ...

    return {
        "model": model,
        "model_dir": model_dir,
        "ckpt_path": ckpt_path,
        "ckpt_step": ckpt_step,
        "family": family,
        "name": name,
    }
