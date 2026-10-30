import os
import random
from collections import deque
import torch
import numpy as np
import nibabel as nib
import torch
import torchvision
import monai.transforms as transforms
import lpips
import wandb 
from matplotlib import pyplot as plt
from einops import rearrange
import re
import time
import datetime
import importlib
from PIL import Image
import torchvision.io as io
from collections import deque, defaultdict
from tqdm import tqdm
from typing import Any
from omegaconf import OmegaConf, DictConfig, ListConfig
import glob
import os
import random
import torch
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from torch.utils.data.dataloader import default_collate


def collate_with_strings(batch):
    """
    Custom collate function that safely batches tensors, numbers, dicts,
    and keeps strings (or other non-tensor objects) as lists.
    """
    if isinstance(batch[0], dict):
        collated = {}
        for key in batch[0]:
            vals = [d[key] for d in batch]
            try:
                collated[key] = default_collate(vals)
            except TypeError:
                # default_collate fails on non-tensor types (like strings)
                collated[key] = vals
        return collated
    else:
        # Fallback: if dataset returns tuples instead of dicts
        try:
            return default_collate(batch)
        except TypeError:
            return batch

def collate_skip_none(batch):
    start_time = time.time()
    # batch is a list of items returned by __getitem__
    # e.g. [sample1, None, sample2, ...]
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None  # in case all items were None
    return default_collate(batch)  # use PyTorch’s default for the rest

def remove_module_prefix(state_dict):
    return {k.replace('module.', ''): v for k, v in state_dict.items()}

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_checkpoint(ckpt_path):
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"No checkpoint found or checkpoint {ckpt_path} does not exist. Starting from scratch.")
        return None

    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    return state

def get_monai_transforms(cfg):
    trfs = []
    for t in cfg:
        transform_cls = instantiate_from_config(t)
        trfs.append(transform_cls)
    trfs = transforms.Compose(trfs)

    return trfs

def get_torchvision_transforms(cfg):
    trfs = []
    for t in cfg:
        transform_cls = instantiate_from_config(t)
        trfs.append(transform_cls)
    trfs = torchvision.transforms.Compose(trfs)

    return trfs

def apply_gpu_transforms_per_sample(x, transform):
    out = []
    for i in range(x.shape[0]):
        t = transform(x[i])
        out.extend(t if isinstance(t, list) else [t])
    return torch.stack(out)

class DummyProfiler:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    def step(self):
        pass
    @property
    def running(self):
        return False
    def stop(self):
        pass

def delete_output_dir(base_output_dir):
    if os.path.exists(base_output_dir):
        print(f"Deleting existing output directory: {base_output_dir}")
        import shutil
        shutil.rmtree(base_output_dir)

def setup_output_dirs(config, output_dir=None):
    if output_dir is None:
        output_dir = Path(os.getcwd())
    else: 
        output_dir = Path(output_dir)
    
    # Create train and media subdirectories
    train_dir = output_dir / "train"
    train_dir.mkdir(exist_ok=True, parents=True)
    
    for dir_name in ["models", "media"]:
        (train_dir / dir_name).mkdir(exist_ok=True)
    
    # Save config
    config_path = output_dir / "model_config.yaml"
    OmegaConf.save(config, str(config_path))
    
    return train_dir

def find_latest_checkpoint(base_output_dir):
    checkpoints = glob.glob(os.path.join(base_output_dir, "train/models/ckpt.*.pt"))
    if not checkpoints:
        print("No checkpoints found in the output directory: ", base_output_dir)
        return None
    # Filter out the best checkpoint and get the latest numbered checkpoint
    checkpoints = [(int(ckpt.split('ckpt.')[1].split('.pt')[0]), ckpt) 
                  for ckpt in checkpoints if 'ckpt.best.pt' not in ckpt]
    print("Found the following checkpoints: ", checkpoints)
    return max(checkpoints, key=lambda x: x[0])[1] if checkpoints else None

def init_optimizers(config, model, loss_fn=None):
    if not hasattr(config, 'optimizer'):
        return None

    if isinstance(config.optimizer, DictConfig) and 'gen' in config.optimizer and 'disc' in config.optimizer:
        gen_params = [p for name, p in model.named_parameters() if 'discriminator' not in name]
        disc_params = list(loss_fn.discriminator.parameters()) if loss_fn and hasattr(loss_fn, 'discriminator') else []
        
        if not disc_params:
            raise ValueError("No discriminator parameters found in loss function")
            
        return {
            'gen': instantiate_from_config(config.optimizer.gen, params=gen_params),
            'disc': instantiate_from_config(config.optimizer.disc, params=disc_params)
        }
    
    return instantiate_from_config(config.optimizer, params=model.parameters())

def init_schedulers(config, optimizer, num_processes=1, epoch_to_steps=None):
    if not hasattr(config, 'scheduler'):
        return None
    
    if optimizer is None:
        return None

    if isinstance(optimizer, dict) and isinstance(config.scheduler, DictConfig):
        return {k: instantiate_from_config(getattr(config.scheduler, k), optimizer=optimizer[k], num_processes=1, epoch_to_steps=epoch_to_steps)
                for k in ['gen', 'disc'] if k in optimizer and hasattr(config.scheduler, k)}
    
    return instantiate_from_config(config.scheduler, optimizer=optimizer, num_processes=1, epoch_to_steps=epoch_to_steps)

def get_obj_from_str(string: str, reload: bool = False):
    module, cls = string.rsplit(".", 1)
    module_imp = importlib.import_module(module)
    if reload:
        importlib.reload(module_imp)
    return getattr(module_imp, cls)

def instantiate_from_config(config, *args, **kwargs):
    if isinstance(config, (DictConfig, ListConfig)):
        config = OmegaConf.to_container(config, resolve=True)

    if config is None:
        return None

    if isinstance(config, list):
        return [instantiate_from_config(item) for item in config]

    if isinstance(config, dict):
        if "_target_" in config:
            cls = get_obj_from_str(config["_target_"])
            init_args = {
                k: instantiate_from_config(v) for k, v in config.items() if k != "_target_"
            }
            return cls(*args, **init_args, **kwargs)
        else:
            return {k: instantiate_from_config(v) for k, v in config.items()}

    return config

def estimate_model_size(model) -> float:
    """
    Estimate the size of a PyTorch model in gigabytes.

    Args:
    model (Union[torch.nn.Module, dict]): The PyTorch model or state_dict to estimate.

    Returns:
    float: Estimated size of the model in gigabytes.
    """
    if isinstance(model, torch.nn.Module):
        state_dict = model.state_dict()
    elif isinstance(model, dict):
        state_dict = model
    else:
        raise ValueError("Input must be a PyTorch model or state_dict")

    param_size = 0
    for param in state_dict.values():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    size_all_gb = size_all_mb / 1024

    return size_all_gb

class Meter:
    def __init__(self):
        self._deque = deque()
        self._count = 0
        self._total = 0.0

    def update(self, value):
        self._deque.append(value)
        self._count += 1
        self._total += value

    def reset(self):
        self._deque.clear()
        self._count = 0
        self._total = 0.0

    @property
    def avg(self):
        if len(self._deque) == 0:
            return 0.0
        d = torch.tensor(list(self._deque), dtype=torch.float)
        return d.mean().item()

    @property
    def std(self):
        if len(self._deque) == 0:
            return 0.0
        d = torch.tensor(list(self._deque), dtype=torch.float)
        return d.std(unbiased=False).item()

    @property
    def max(self):
        if len(self._deque) == 0:
            return 0.0
        d = torch.tensor(list(self._deque), dtype=torch.float)
        return d.max().item()

    @property
    def min(self):
        if len(self._deque) == 0:
            return 0.0
        d = torch.tensor(list(self._deque), dtype=torch.float)
        return d.min().item()
    
    @property
    def median(self):
        if len(self._deque) == 0:
            return 0.0
        d = torch.tensor(list(self._deque), dtype=torch.float)
        return d.median().item()

    @property
    def latest(self):
        return self._deque[-1] if len(self._deque) > 0 else None

class IterMeter:
    def __init__(self):
        self._deque = deque()
        self._count = 0
        self._total = 0.0

    def update(self, value):
        self._deque.append(value)
        self._count += 1
        self._total += value

    def reset(self):
        self._deque.clear()
        self._count = 0
        self._total = 0.0

    @property
    def avg(self):
        if len(self._deque) == 0:
            return 0.0
        d = torch.tensor(list(self._deque), dtype=torch.float)
        return d.mean().item()

    @property
    def latest(self):
        return self._deque[-1] if len(self._deque) > 0 else None

class MultiMeter:
    def __init__(self):
        self.meters = defaultdict(IterMeter)
        self.start_time = time.time()

    def update(self, name, value):
        self.meters[name].update(value)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()
        self.start_time = time.time()

    def get_average(self, name):
        return self.meters[name].avg

    def get_latest(self, name=None):
        if name is None:
            return {name: meter.latest for name, meter in self.meters.items()}
        return self.meters[name].latest

    def get_averages(self):
        return {name: meter.avg for name, meter in self.meters.items()}

    def get_latest_values(self):
        return {name: meter.latest for name, meter in self.meters.items()}

    def __str__(self):
        return " | ".join([f"{name}: {meter.avg:.4f}" for name, meter in self.meters.items()])  

    def log_every(self, iterable, total=None, disable=False):
        MB = 1024.0 * 1024.0
        pbar = tqdm(iterable, total=total, disable=disable)
        end_time = time.time()
        for obj in pbar:
            start_data = time.time()
            yield obj  # gives batch to training loop
            self.update("t_batch", start_data - end_time)  # only time between iterations (loader wait)
            
            self.update("t_iter", time.time() - start_data)
            self.update("iter_memory", torch.cuda.max_memory_allocated() / MB)
            torch.cuda.reset_peak_memory_stats()

            pbar.set_postfix({name: f"{meter.latest:.6f}" for name, meter in self.meters.items() if meter.latest is not None})
            end_time = time.time()
        pbar.close()

def build_dataset_file_list(
    root_dir: str,
    file_names: list[str],
    sbj_include: str | None = None,
    sbj_exclude: str | None = None,
    n_subjects: int | None = None,
) -> list[dict]:
    """
    Build a MONAI-compatible list of file dictionaries from a structured dataset directory.

    Args:
        root_dir (str): Root directory containing subject subfolders.
        file_names (list[str]): List of filenames to look for within each subject folder.
                                Example: ["inp.nii.gz", "wat.nii.gz"]
        sbj_include (str | None): Optional path to a file containing specific subject IDs to include.
        sbj_exclude (str | None): Optional path to a file containing specific subject IDs to exclude.
        n_subjects (int | None): Optional maximum number of subjects to load (for debugging / limiting dataset).

    Returns:
        list[dict]: A list of dictionaries, each like {"image": <file_path>}.

    Example:
        >>> train_files = build_dataset_file_list(
        ...     root_dir="/data/whole_body/nifti",
        ...     file_names=["inp.nii.gz"],
        ...     sbj_include="include_subjects.txt"
        ... )
        >>> train_files[0]
        {'image': '/data/whole_body/nifti/2033420/inp.nii.gz'}
    """
    all_files = []
    if sbj_exclude is not None and os.path.isfile(sbj_exclude):
        with open(sbj_exclude, "r") as f:
            subjects_exclude = [line.strip() for line in f if line.strip()]

    # --- Load subject IDs ---
    if sbj_include is not None and os.path.isfile(sbj_include):
        with open(sbj_include, "r") as f:
            subjects = [line.strip() for line in f if line.strip()]
    else:
        subjects = sorted(os.listdir(root_dir))

    if sbj_exclude is not None and os.path.isfile(sbj_exclude):
        subjects = [sub for sub in subjects if sub not in subjects_exclude]

    # --- Iterate through subjects ---
    for subject in tqdm(subjects, desc="Building file list"):
        subject_dir = os.path.join(root_dir, subject)
        if not os.path.isdir(subject_dir):
            continue

        # --- Limit number of subjects ---
        if n_subjects is not None and len(all_files) >= n_subjects:
            break

        # --- Collect available files for each subject ---
        for file_name in file_names:
            file_path = os.path.join(subject_dir, file_name)
            if os.path.isfile(file_path):
                all_files.append({
                    "image": file_path,
                    "sbj": subject,
                    "file_name": os.path.splitext(os.path.basename(file_name))[0],
                })

    all_files = sorted(all_files, key=lambda x: x["image"])
    assert len(all_files) > 0, f"No files found for {root_dir}!"

    print(f"Found {len(all_files)} files in {root_dir}")
    print(all_files[:10])
    return all_files

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Add CUDA-specific seeding for more deterministic behavior
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def training_reproducibility_cudnn():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

def get_model_hash(model):
    import hashlib
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    for b in model.buffers():
        h.update(b.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def init_from_ckpt(self, path, weights_only=False):
        # Load the checkpoint
        try:
            sd = torch.load(path, map_location="cpu", weights_only=weights_only)['state_dict']
        except:
            try: 
                sd = torch.load(path, map_location="cpu", weights_only=weights_only)['model']
            except:
                sd = torch.load(path, map_location="cpu", weights_only=weights_only)

        # Remove '.module' prefix from keys if present
        new_sd = {}
        for k, v in sd.items():
            if "loss" in k: continue
            new_key = k.replace("module.", "") if k.startswith("module.") else k
            new_sd[new_key] = v
        
        # Try to load the state dictionary, but allow for dimension mismatches
        try:
            msg = self.load_state_dict(new_sd, strict=True)
        except RuntimeError as e:
            # If error is due to dimension mismatch (2D vs 3D), print warning and skip loading affected layers
            print(f"Warning: {str(e)}")
            print("This might be due to loading 2D weights into 3D model or vice versa.")
            print("Loading weights with strict=False to skip incompatible layers.")
            msg = self.load_state_dict(new_sd, strict=False)

        # Clean up memory
        torch.cuda.empty_cache()

        # Print loading information
        print(f"Loading pre-trained {self.__class__.__name__}")
        print("Missing keys:")
        print(msg.missing_keys)
        print("Unexpected keys:")
        print(msg.unexpected_keys)
        print(f"Restored from {path}")


class ToDisk:
    def __init__(self, formats=["gif"]):
        self.formats = formats

    def to_disk(self, path, vid_img: torch.Tensor, normalize=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        for fmt in self.formats:
            fmt_path = path + "." + fmt
            if fmt == "gif":
                self.vid_or_image_to_disk(fmt_path, vid_img, normalize)
            elif fmt == "nii.gz":
                self.vid_or_image_to_nifti(fmt_path, vid_img, normalize)
            elif fmt == "png":
                self.vid_or_image_to_png(fmt_path, vid_img, normalize)
            else:
                raise Exception(f"format {fmt} not supported")

    def _extend_vid_or_img(self, vid_img):
        if len(vid_img.shape) == 3:
            vid_img = vid_img.unsqueeze(-3)
        if len(vid_img.shape) == 4:
            vid_img = vid_img.unsqueeze(-3)
        if len(vid_img.shape) == 5:
            if vid_img.shape[1] == 1:
                vid_img = vid_img.repeat(1, 3, 1, 1, 1)
        else: 
            raise Exception("given shape of data cannot be handled")
        return vid_img

    def vid_or_image_to_png(self, path, vid_img: torch.Tensor, normalize=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        assert len(vid_img.shape) < 6, f"Cannot process shape {vid_img.shape}"
        if normalize:
            vid_img = (vid_img + 1) / 2

        if len(vid_img.shape) == 5:
            vid_img = rearrange(vid_img, "b c t h w -> (b t) c h w") #put time into batch dimension
        if len(vid_img.shape) == 4:
            vid_img = rearrange(vid_img, "b c h w -> c (b h) w") #put batch into height
        

        io.write_png((vid_img.cpu()*255).byte(), path)

    def videos_to_disk(self, path, vid):
        vid = rearrange(vid, 'b c t h w -> t (b h) w c').clamp(0,1).cpu().detach().numpy()
        vid = vid * 255    
        videos = [Image.fromarray(image) for image in vid.astype(np.uint8)]
        videos[0].save(path, save_all=True, append_images=videos[1:], duration=1000/8, loop=0)


    def vid_or_image_to_disk(self, path, vid_img: torch.Tensor, normalize=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if normalize:
            vid_img = (vid_img + 1) / 2

        vid_img = self._extend_vid_or_img(vid_img)
        self.videos_to_disk(path, vid_img)


    def vid_or_image_to_nifti(self, path, vid_img: torch.Tensor, normalize=False):
        if not path.endswith('.nii.gz'):
            path = os.path.splitext(path)[0] + '.nii.gz'
        os.makedirs(os.path.dirname(path), exist_ok=True)

        vid_img = vid_img[0,0,...]
        if normalize:
            vid_img = (vid_img + 1) / 2

        vid_img = vid_img.float().clamp(0,1).cpu().detach().numpy()
        img = nib.Nifti1Image(vid_img, np.eye(4))
        nib.save(img, path)
        
        
def delay2str(t):
    """
    Convert a given time in seconds to a formatted string representation with days, hours, minutes, and seconds.
    
    Parameters:
        t (int): The time in seconds to be converted.
    
    Returns:
        str: The formatted string representation of the time including days, hours, minutes, and seconds.
    """
    t = int(t)
    secs = t%60
    mins = (t//60)%60
    hours = (t//3600)%24
    days = t//86400
    string = f"{secs}s"
    if mins:
        string = f"{mins}m {string}"
    if hours:
        string = f"{hours}h {string}"
    if days:
        string = f"{days}d {string}"
    return string



def get_torchvision_unnormalize(mean, std):
    """
    Get unnormalize function. reference: https://github.com/pytorch/vision/issues/528

    Args:
        mean, std (list): Normalization parameters (RGB)

    Returns:
        unnormalize (torchvision.transforms.Normalize): Unnormalize function.
    """
    assert len(mean) == 3
    assert len(std) == 3
    mean = torch.tensor(mean, dtype=torch.float32)
    std = torch.tensor(std, dtype=torch.float32)

    unnormalize = torchvision.transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    return unnormalize


def extract_seven_concurrent_numbers(text):
    pattern = r'\b\d{7}\b'
    seven_numbers = re.findall(pattern, text)
    return seven_numbers

def get_plane_from_normal_and_position(volume, normal_vectors=None, position_vectors=None, device=torch.device("cpu")):
    assert len(volume.shape) == 4, "Volume has to have 4 dimensions including batch dimension"
    b, h, w, s = volume.shape
    if normal_vectors is None:
        normal_vectors = torch.rand((b, 3), device=device) *2 - 1 #[-1, 1]
        normal_vectors = normal_vectors / torch.norm(normal_vectors, dim=1, keepdim=True)

    if position_vectors is None:
        position_vectors = torch.tensor([[h//2, w//2, s//2],]).repeat(b, 1).to(device)

    # Create a zero-centered grid for the new plane
    u = torch.linspace(-h//2, h//2, h)
    v = torch.linspace(-w//2, w//2, w)
    U, V = torch.meshgrid(u, v, indexing='ij')
    U = U.to(device)
    V = V.to(device)

    # Calculate basis vectors for the planes
    u_axes = torch.zeros_like(normal_vectors)
    v_axes = torch.zeros_like(normal_vectors)

    for i in range(b):
        if torch.allclose(normal_vectors[i], torch.tensor([0.0, 0.0, 1.0], device=device)) or torch.allclose(normal_vectors[i], torch.tensor([0.0, 0.0, -1.0], device=device)):
            u_axes[i] = torch.tensor([1.0, 0.0, 0.0])
            v_axes[i] = torch.tensor([0.0, 1.0, 0.0])
        else:
            u_axes[i] = torch.linalg.cross(normal_vectors[i], torch.tensor([0.0, 0.0, 1.0], device=device))
            u_axes[i] = u_axes[i] / torch.norm(u_axes[i])
            v_axes[i] = torch.linalg.cross(normal_vectors[i], u_axes[i])

    # Generate points on the planes, shifted by the position vectors
    X = position_vectors[:, 0, None, None] + U * u_axes[:, 0, None, None] + V * v_axes[:, 0, None, None]
    Y = position_vectors[:, 1, None, None] + U * u_axes[:, 1, None, None] + V * v_axes[:, 1, None, None]
    Z = position_vectors[:, 2, None, None] + U * u_axes[:, 2, None, None] + V * v_axes[:, 2, None, None]

    # Prepare points for interpolation
    points = torch.stack((X, Y, Z), dim=-1)

    for point in points:
        # Normalize grid coordinates to [-1, 1]
        point[:, :, 0] = (point[:, :, 0] - h/2) / (h/2)
        point[:, :, 1] = (point[:, :, 1] - w/2) / (w/2)
        point[:, :, 2] = (point[:, :, 2] - s/2) / (s/2)

    # Reshape grid for grid_sample
    points = points.unsqueeze(1)  # Add channel dimension

    # Reshape volume for grid_sample
    volume = volume.unsqueeze(1)  # Add channel dimension

    # Interpolate values
    plane_values = torch.nn.functional.grid_sample(
        volume.to(device),
        points.to(device),
        mode='bilinear',
        padding_mode='zeros',
        align_corners=True
    ).squeeze(1).squeeze(1)  # Remove channel dimension
    return plane_values, normal_vectors, position_vectors