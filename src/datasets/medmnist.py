import torch
from torch.utils.data import Dataset
import numpy as np
import medmnist
from medmnist import INFO
from torch.utils.data import Dataset
import torchvision.transforms as transforms

two_d_datasets = [
    name for name, info in INFO.items()
    if "3d" not in name
]

three_d_datasets = [
    name for name, info in INFO.items()
    if "3d" in name
]

def load_dataset(root, name, split, transform=None, download=True, size=224):
    info = INFO[name]
    DataClass = getattr(medmnist, info["python_class"])
    return DataClass(root=root, split=split, transform=transform, download=download, as_rgb=True, size=size)

class MergedMedMNIST(Dataset):
    # Use the canonical/standard order for dataset IDs
    canonical_order = [
        "pathmnist",
        "chestmnist",
        "dermamnist",
        "octmnist",
        "pneumoniamnist",
        "retinamnist",
        "breastmnist",
        "bloodmnist",
        "tissuemnist",
        "organamnist",
        "organcmnist",
        "organsmnist"
    ]

    # Map from dataset name to its fixed ID, e.g. "retinamnist"->5
    canonical_id_map = {name: idx for idx, name in enumerate(canonical_order)}

    def __init__(self, root, split="train", include_dataset_id=True, transform=None, download=True, size=224, ds_names=two_d_datasets):
        self.datasets = []
        self.index_map = []   # (dataset_id, index_inside_dataset)
        self.include_dataset_id = include_dataset_id
        self.split = split
        self.ds_name_to_dataset_id = {}
        self.ds_id_to_flag = {}
        # Load datasets (lightweight: only metadata + mmap arrays)
        for name in ds_names:
            # Always use canonical/fixed ID
            dataset_id = self.canonical_id_map.get(name, None)
            if dataset_id is None:
                raise ValueError(f"Dataset name '{name}' is not in canonical_order list for MedMNIST merged loader.")

            print(f"Loading dataset: {name} (fixed ID={dataset_id})")
            ds = load_dataset(root, name, split, transform, download, size)
            self.datasets.append(ds)
            self.ds_name_to_dataset_id[name] = dataset_id
            self.ds_id_to_flag[dataset_id] = ds.flag

            # Build mapping: one entry per sample
            for i in range(len(ds)):
                self.index_map.append((dataset_id, i, len(self.datasets) - 1))  # dataset_id, sample_index, which self.datasets index

        # Since self.datasets order may not be canonical (user can re-order ds_names arbitrarily), we must store mapping from merged_list_idx to canonical_id
        # index_map now stores (canonical_id, inside_ds_idx, local_ds_array_id)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        dataset_id, inner_idx, local_ds_array_idx = self.index_map[idx]
        ds = self.datasets[local_ds_array_idx]

        img, label = ds[inner_idx]  # MedMNIST handles lazy loading

        # chestmnist is multiclass we just neglect it 
        if ds.flag == "chestmnist":
            label = np.array([0])

        file_name = f"{ds.flag}_{self.split}_{inner_idx}"

        if self.include_dataset_id:
            return {"image": img, "label": label, "dataset_id": dataset_id, "sbj": idx, "file_name": file_name}
        else:
            return {"image": img, "label": label, "sbj": idx, "file_name": file_name}