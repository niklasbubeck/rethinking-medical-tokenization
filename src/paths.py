"""Default locations, overridable through environment variables.

All pipeline scripts resolve their default paths here so the repository runs
anywhere without editing code. Every value can also be overridden per run with
the corresponding CLI flag.

  RMT_BASE_DIR      root holding `outputs/<project>/<model>/` (checkpoints,
                    evaluation dumps, generated samples). Default: ./
  RMT_PROJECT       name of the `<project>` sub-folder.      Default: tokenizers
  MEDMNIST_ROOT     MedMNIST+ `.npz` files (`<dataset>_224.npz`).
                    Default: ./data/medmnist
  MEDMNIST_WEIGHTS  official MedMNIST ResNet baselines used as the FMD feature
                    extractor, laid out as `<dataset>/resnet18_224_1.pth`.
                    Default: ./data/medmnist_weights
  SSCD_PATH         SSCD copy-detection torchscript model.
                    Default: ./data/models/sscd_disc_mixup.torchscript.pt
  TORCH_HUB_DIR     torch.hub cache for DINOv2 (None = torch default).
"""
import os

BASE_DIR = os.environ.get("RMT_BASE_DIR", ".")
PROJECT = os.environ.get("RMT_PROJECT", "tokenizers")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MEDMNIST_ROOT = os.environ.get("MEDMNIST_ROOT", os.path.join("data", "medmnist"))
MEDMNIST_WEIGHTS = os.environ.get("MEDMNIST_WEIGHTS", os.path.join("data", "medmnist_weights"))
SSCD_PATH = os.environ.get(
    "SSCD_PATH", os.path.join("data", "models", "sscd_disc_mixup.torchscript.pt"))
TORCH_HUB_DIR = os.environ.get("TORCH_HUB_DIR") or None
