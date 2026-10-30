from dataclasses import dataclass
import torchvision.transforms as T
from torch.utils.data import DataLoader

from src.datasets.medmnist import load_dataset, two_d_datasets


@dataclass
class DatasetSpec:
    name: str
    n_classes: int
    task_type: str   # "multiclass" | "multilabel" | "ordinal"
    primary_metric: str  # which metric to early-stop / rank by


# Hand-curated mapping per the user's table.
# task_type drives loss + metrics dispatch in probe.py.
DATASET_SPECS = {
    "pathmnist":      DatasetSpec("pathmnist",      9,  "multiclass", "auroc_macro"),
    "chestmnist":     DatasetSpec("chestmnist",     14, "multilabel", "auroc_macro"),
    "dermamnist":     DatasetSpec("dermamnist",     7,  "multiclass", "auroc_macro"),
    "octmnist":       DatasetSpec("octmnist",       4,  "multiclass", "auroc_macro"),
    "pneumoniamnist": DatasetSpec("pneumoniamnist", 2,  "multiclass", "auroc_macro"),
    "retinamnist":    DatasetSpec("retinamnist",    5,  "ordinal",    "qwk"),
    "breastmnist":    DatasetSpec("breastmnist",    2,  "multiclass", "auroc_macro"),
    "bloodmnist":     DatasetSpec("bloodmnist",     8,  "multiclass", "auroc_macro"),
    "tissuemnist":    DatasetSpec("tissuemnist",    8,  "multiclass", "auroc_macro"),
    "organamnist":    DatasetSpec("organamnist",    11, "multiclass", "auroc_macro"),
    "organcmnist":    DatasetSpec("organcmnist",    11, "multiclass", "auroc_macro"),
    "organsmnist":    DatasetSpec("organsmnist",    11, "multiclass", "auroc_macro"),
}

ALL_2D = [n for n in DATASET_SPECS if n in two_d_datasets]


def get_dataset_spec(name):
    if name not in DATASET_SPECS:
        raise KeyError(f"Unknown dataset '{name}'. Known: {sorted(DATASET_SPECS)}")
    return DATASET_SPECS[name]


def _make_transform(image_size):
    # Plain ToTensor, no augmentation. Linear probing convention: deterministic features.
    # MedMNIST already provides resized arrays for size=224 etc, so no Resize needed.
    return T.ToTensor()


def build_probe_loaders(name, root, batch_size=256, num_workers=8, image_size=224, download=False):
    """
    Build train/val/test DataLoaders for one MedMNIST dataset.

    Returns: (loaders dict {split: DataLoader}, DatasetSpec).
    """
    spec = get_dataset_spec(name)
    transform = _make_transform(image_size)
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = load_dataset(root, name, split, transform=transform, download=download, size=image_size)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,                  # deterministic for caching
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            drop_last=False,
        )
    return loaders, spec
