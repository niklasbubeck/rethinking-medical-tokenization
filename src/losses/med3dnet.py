import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import time

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]

URL_MAP = "https://drive.google.com/file/d/13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG/view"

# Map model constructor names to pretrained weight locations.
# Update these URLs/IDs as you obtain more checkpoints.
PRETRAINED_URLS = {
    # If a specific URL isn't provided for a model, we will fall back to URL_MAP.
    'resnet10': URL_MAP,
    # 'resnet18': 'https://.../resnet18_med3d.pth',
    # 'resnet34': 'https://.../resnet34_med3d.pth',
    # 'resnet50': 'https://.../resnet50_med3d.pth',
    # 'resnet101': 'https://.../resnet101_med3d.pth',
    # 'resnet152': 'https://.../resnet152_med3d.pth',
    # 'resnet200': 'https://.../resnet200_med3d.pth',
}


def _is_google_drive_url(url: str) -> bool:
    return 'drive.google.com' in url


def _extract_gdrive_id(url: str) -> str:
    # Supports formats like:
    # https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
    # https://drive.google.com/uc?id=<FILE_ID>
    # https://drive.google.com/drive/folders/<FOLDER_ID>
    if '/file/d/' in url:
        try:
            return url.split('/file/d/')[1].split('/')[0]
        except Exception:
            return ''
    if '/folders/' in url:
        try:
            return url.split('/folders/')[1].split('?')[0].split('/')[0]
        except Exception:
            return ''
    if 'id=' in url:
        try:
            return url.split('id=')[1].split('&')[0]
        except Exception:
            return ''
    return ''


def _download_with_gdown(file_id_or_url: str, output_path: str) -> None:
    try:
        import gdown  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            'Downloading from Google Drive requires the "gdown" package. '
            'Install it with: pip install gdown'
        ) from exc
    url = file_id_or_url if file_id_or_url.startswith('http') else f'https://drive.google.com/uc?id={file_id_or_url}'
    gdown.download(url, output_path, quiet=False)


def _download_generic(url: str, output_path: str) -> None:
    from torch.hub import download_url_to_file
    download_url_to_file(url, output_path, progress=True)


def _download_weights(url: str, output_path: str) -> None:
    if _is_google_drive_url(url):
        file_id = _extract_gdrive_id(url)
        _download_with_gdown(file_id or url, output_path)
    else:
        _download_generic(url, output_path)


def _maybe_extract_archive(archive_path: str, extract_dir: str) -> str:
    import os
    import zipfile
    import tarfile
    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(extract_dir)
            return extract_dir
        if tarfile.is_tarfile(archive_path):
            with tarfile.open(archive_path, 'r:*') as tf:
                tf.extractall(extract_dir)
            return extract_dir
    except Exception:
        pass
    return ''


def _find_checkpoint_in_dir(search_dir: str, model_name: str) -> str:
    import os
    matches = []
    for root, _dirs, files in os.walk(search_dir):
        for fname in files:
            lower = fname.lower()
            if lower.endswith(('.pth', '.pt', '.ckpt', '.bin')):
                matches.append(os.path.join(root, fname))
    if not matches:
        return ''

    name_mapping = {
        'resnet10': 'resnet_10.pth',
        'resnet18': 'resnet_18.pth',
        'resnet34': 'resnet_34.pth',
        'resnet50': 'resnet_50.pth',
        'resnet101': 'resnet_101.pth',
        'resnet152': 'resnet_152.pth',
        'resnet200': 'resnet_200.pth'
    }

    expected = name_mapping.get(model_name.lower())
    if expected:
        exact = [p for p in matches if os.path.basename(p).lower() == expected]
        if exact:
            return exact[0]

    contains = [p for p in matches if model_name.lower() in os.path.basename(p).lower()]
    if contains:
        return contains[0]

    # last resort: prefer largest file
    return max(matches, key=lambda p: os.path.getsize(p))


def _ensure_weights(model_name: str, url: str, weights_dir: str | None = None) -> str:
    import os
    base_dir = weights_dir or os.path.expanduser('~/.cache/med3d')
    target_dir = os.path.join(base_dir, 'resnets')
    os.makedirs(target_dir, exist_ok=True)

    # 1) If we already have the correct file in target_dir (or its subdirs), return it
    existing = _find_checkpoint_in_dir(target_dir, model_name)
    if existing:
        return existing

    # 2) Download to a temp file then try to extract into target_dir
    tmp_path = os.path.join(target_dir, 'download.tmp')
    _download_weights(url, tmp_path)

    extracted = _maybe_extract_archive(tmp_path, target_dir)
    if extracted:
        found = _find_checkpoint_in_dir(target_dir, model_name)
        if found:
            return found

    # 3) If the download itself is a checkpoint, move/rename it to expected location
    import shutil
    if os.path.splitext(tmp_path)[1].lower() in ('.pth', '.pt', '.ckpt', '.bin'):
        name_mapping = {
            'resnet10': 'resnet_10.pth',
            'resnet18': 'resnet_18.pth',
            'resnet34': 'resnet_34.pth',
            'resnet50': 'resnet_50.pth',
            'resnet101': 'resnet_101.pth',
            'resnet152': 'resnet_152.pth',
            'resnet200': 'resnet_200.pth'
        }
        dst = os.path.join(target_dir, name_mapping.get(model_name.lower(), os.path.basename(tmp_path)))
        try:
            shutil.move(tmp_path, dst)
        except Exception:
            dst = tmp_path
        return dst

    # 4) Fallback: keep file as-is and return best match under target_dir or the tmp file
    found = _find_checkpoint_in_dir(target_dir, model_name)
    if found:
        return found
    return tmp_path


def load_pretrained_med3d(
    model_name: str,
    *,
    shortcut_type: str = 'B',
    no_cuda: bool = False,
    weights_dir: str | None = None,
    map_location: str | torch.device = 'cpu',
    strict: bool = False,
):
    """
    Instantiate a Med3D model by name and automatically download and load
    its pretrained weights.

    Parameters
    - model_name: One of {'resnet10','resnet18','resnet34','resnet50','resnet101','resnet152','resnet200'}
    - sample_input_D/H/W, num_seg_classes, shortcut_type, no_cuda: forwarded to the constructor
    - weights_dir: optional directory to cache downloaded weights (default: ~/.cache/med3d)
    - map_location: where to load the state dict
    - strict: pass-through to model.load_state_dict

    Returns
    - model: torch.nn.Module with weights loaded
    """
    name = model_name.lower()
    if name not in __all__:
        raise ValueError(f'Unknown Med3D model name: {model_name}. Expected one of {list(__all__)}')

    constructor = globals().get(name)
    if constructor is None:
        raise ValueError(f'Model constructor not found for name: {model_name}')

    model = constructor(
        shortcut_type=shortcut_type,
        no_cuda=no_cuda,
    )

    url = PRETRAINED_URLS.get(name, URL_MAP)
    if not url:
        return model

    weights_path = _ensure_weights(name, url, weights_dir)
    print(f"Loading 3D LPIPSweights path from: {weights_path}")

    state_dict = torch.load(weights_path, map_location=map_location, weights_only=True)
    # Some checkpoints may store under key 'state_dict'
    if isinstance(state_dict, dict) and 'state_dict' in state_dict and isinstance(state_dict['state_dict'], dict):
        state_dict = state_dict['state_dict']

    # If keys are prefixed (e.g., 'module.'), try to sanitize
    if isinstance(state_dict, dict):
        sample_key = next(iter(state_dict.keys()))
        if sample_key.startswith('module.'):
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if strict is False and (missing or unexpected):
        # Non-strict load; informatively print in logs
        try:
            import logging
            logging.getLogger(__name__).warning(
                'Loaded Med3D weights with non-strict matching. Missing: %s | Unexpected: %s',
                missing, unexpected
            )
        except Exception:
            pass
    for param in model.parameters():
        param.requires_grad = False
    return model


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = torch.cat([out.data, zero_pads], dim=1)
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Flatten(torch.nn.Module):
    def forward(self, inp):
        return inp.view(inp.size(0), -1)


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 no_cuda=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        self.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), Flatten())

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_seg(x)

        return x


def resnet10(**kwargs):
    """Constructs a ResNet-10 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-152 model.
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-200 model.
    """
    model = ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model


def normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    """Channel-wise L2 normalize"""
    return x / (torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True)) + 1e-10)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3, 4], keepdim=keepdim)


class Normalize(nn.Module):
    def __init__(self, mean=0.5, std=0.5):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        return (x - self.mean) / (self.std + 1e-6)


class NetLinLayer3D(nn.Module):
    """LPIPS linear model per feature map (1×1×1 conv)"""
    def __init__(self, chn_in, use_dropout=False):
        super().__init__()
        layers = []
        if use_dropout:
            layers.append(nn.Dropout())
        layers.append(nn.Conv3d(chn_in, 1, kernel_size=1, stride=1, padding=0, bias=False))
        self.model = nn.Sequential(*layers)


class LPIPS3D(nn.Module):
    """True LPIPS-style perceptual similarity for 3D volumes."""
    FEATURE_CHANNELS = {
        "resnet10": [64, 128, 256, 512],
        "resnet18": [64, 128, 256, 512],
        "resnet34": [64, 128, 256, 512],
        "resnet50": [256, 512, 1024, 2048],
        "resnet101": [256, 512, 1024, 2048],
        "resnet152": [256, 512, 1024, 2048],
        "resnet200": [256, 512, 1024, 2048],
    }
    def __init__(
        self,
        model_name="resnet10",
        device="cpu",
        weights_dir=None,
        use_dropout=True,
    ):
        super().__init__()
        self.device = torch.device(device)

        self.FEATURE_CHANNELS = self.FEATURE_CHANNELS[model_name]
        # Load pretrained Med3D as backbone
        net = load_pretrained_med3d(
            model_name,
            map_location=self.device,
            weights_dir=weights_dir,
            strict=False,
        ).to(self.device)

        # Extract features from blocks 1–4 only
        self.features = nn.ModuleList([
            nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1),
            nn.Sequential(net.layer2),
            nn.Sequential(net.layer3),
            nn.Sequential(net.layer4),
        ])

        # LPIPS learned weights for each feature map
        self.lins = nn.ModuleList([
            NetLinLayer3D(c, use_dropout=use_dropout).to(self.device)
            for c in self.FEATURE_CHANNELS
        ])

        self.normalize = Normalize().to(self.device)

        # Freeze both backbone + linear layer weights
        for p in self.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        x1, x2 shape: (B, 1, D, H, W)
        Returns: LPIPS distance per sample (B,)
        """
        start_time = time.time()
        x1 = self.normalize(x1.to(self.device))
        x2 = self.normalize(x2.to(self.device))

        outs0, outs1 = [], []
        f1, f2 = x1, x2

        # Extract feature maps from each block
        for i, block in enumerate(self.features):
            f1 = block(f1)
            f2 = block(f2)
            outs0.append(normalize_tensor(f1))
            outs1.append(normalize_tensor(f2))

        # Compute LPIPS per-layer similarity score
        diffs = [(o0 - o1) ** 2 for o0, o1 in zip(outs0, outs1)]

        res = [
            spatial_average(self.lins[i].model(diffs[i]), keepdim=False)
            for i in range(len(diffs))
        ]

        # Weighted sum of feature distances
        val = res[0]
        for r in res[1:]:
            val += r

        end_time = time.time()
        print(f"LPIPS3D time: {end_time - start_time}")
        return val  # shape: (B,)
