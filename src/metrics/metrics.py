from abc import abstractmethod
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import lpips
import sys 
import os
from math import exp
from monai.losses import DiceLoss
from monai.metrics import HausdorffDistanceMetric as HausdorffDistance
from monai.metrics import SurfaceDistanceMetric as SurfaceDistance
from monai.metrics.regression import SSIMMetric as SSIM
from monai.networks.utils import one_hot
from src.accelerator import AccelerateParent

import lpips
import torch
import traceback
from ..utils import instantiate_from_config
import torch_fidelity

class CompositeMetrics():
    """
    Improved CompositeMetrics class for clearer metric evaluation.
    """
    def __init__(self, metrics: list):
        self.tasks = metrics

    def compute_image_metrics(self, gt, pred, mask=None):
        """Computes image-based metrics (LPIPS, MSE, MAE, SSIM, PSNR)."""
        results = {}
        for task in self.tasks:
            if isinstance(task, (LPIPSMetric, MSEMetric, MAEMetric, SSIMMetric, PSNRMetric)):
                results[task.__class__.__name__] = task(gt, pred, mask)
        return results

    def compute_segmentation_metrics(self, gt, pred, mask=None):
        """Computes segmentation metrics (Dice, Hausdorff, Surface Distance)."""
        # if mask is not None:
        #     print("SEGMENTATION METRICS DO NOT SUPPORT MASKING YET")
        results = {}
        for task in self.tasks:
            if isinstance(task, (DiceMetric, HausdorffMetric, SurfaceDistanceMetric, SymmetricSurfaceDistanceMetric)):
                results.update(task(gt, pred))
        return results

    def compute_generation_metrics(self, fake_path, real_path=None):
        """Computes generation metrics (FID, IS, etc.)."""
        results = {}
        for task in self.tasks:
            if isinstance(task, GenerationMetric):
                results.update(task(fake_path, real_path))
        return results

    def compute_all_metrics(self, gt=None, pred=None, mask=None, fake_path=None, one_hot_enc=True, num_classes=4, apply_model=False):
        """Computes all available metrics based on the provided data."""
        results = {}
        if gt is not None and pred is not None:
            results.update(self.compute_image_metrics(gt, pred, mask))
            results.update(self.compute_segmentation_metrics(gt, pred, one_hot_enc, num_classes, apply_model))
        if fake_path is not None:
            results.update(self.compute_generation_metrics(fake_path))
        return results

    @staticmethod
    def _get_evaluation_tasks(cfg):
        tasks = []
        try: 
            for t in cfg:
                task_cls = instantiate_from_config(t)
                tasks.append(task_cls)
        except Exception as e : 
            print(f"Error loading tasks: {e}")
            print(traceback.format_exc())
        return tasks
    
    
class GenerationMetric:
    def __init__(self, gen_cfg):
        self.gen_cfg=gen_cfg

    def __call__(self, fake_path, real_path=None):
        if self.gen_cfg["input2"]  is None:
            self.gen_cfg["input2"] = real_path

        # torch-fidelity raises "At least one metric must be specified" when all
        # metrics are disabled. In that mode we only dump samples to disk (FID is
        # computed offline), so skip the metric call and return nothing.
        if not any(bool(self.gen_cfg.get(k, False)) for k in ("isc", "fid", "kid", "prc")):
            return {}

        return torch_fidelity.calculate_metrics(
            input1=fake_path,
            **self.gen_cfg
        )

class SegmentationMetric(AccelerateParent):
    def __init__(self, seg_model_cfg=None, one_hot_enc=True, num_classes=4, apply_model=True):
        self.seg_model = instantiate_from_config(seg_model_cfg).to(self.device) if seg_model_cfg else None
        self.one_hot_enc = one_hot_enc
        self.num_classes = num_classes
        self.apply_model = apply_model

    def preprocess(self, gt, pred, one_hot_enc=True, num_classes=4, apply_model=True):
        if apply_model and self.seg_model:
            pred = self.seg_model(pred)
        if one_hot_enc:
            gt = one_hot(gt, num_classes=num_classes)
            pred = one_hot(pred, num_classes=num_classes)
        return gt, pred

    def __call__(self, gt, pred):
        gt, pred = self.preprocess(gt, pred, self.one_hot_enc, self.num_classes, self.apply_model)
        metric = self.compute_metric(gt, pred)
        result = {}
        for i in range(metric.shape[1]):
            result[f"{self.__class__.__name__}_{i}"] = metric[:, i]   #mean only to get rid of dimensions
        result[f"{self.__class__.__name__}"] = metric.mean(1)
        return result

    @abstractmethod
    def compute_metric(self, gt, pred):
        pass


class DiceMetric(SegmentationMetric):
    def __init__(self, seg_model_cfg=None, *args, **kwargs):
        super().__init__(seg_model_cfg)
        self.dice = DiceLoss(*args, **kwargs, reduction='none')
    
    def compute_metric(self, gt, pred):
        loss = self.dice(gt, pred)
        metric = 1 - loss
        return metric.view(metric.shape[0], -1)  # somehow it returns [bs, num_classes, 1, 1] ??? so view it!!!


class HausdorffMetric(SegmentationMetric):
    def __init__(self, seg_model_cfg=None, *args, **kwargs):
        super().__init__(seg_model_cfg)
        self.hd = HausdorffDistance(*args, **kwargs, reduction='none')
    
    def compute_metric(self, gt, pred):
        return self.hd(gt, pred)
    
class SurfaceDistanceMetric(SegmentationMetric):
    def __init__(self, seg_model_cfg=None, *args, **kwargs):
        super().__init__(seg_model_cfg)
        self.sd = SurfaceDistance(*args, **kwargs, reduction='none')
    
    def compute_metric(self, gt, pred):
        return self.sd(gt, pred)

class SymmetricSurfaceDistanceMetric(SegmentationMetric):
    def __init__(self, seg_model_cfg=None, *args, **kwargs):
        super().__init__(seg_model_cfg)
        self.sd = SurfaceDistance(*args, symmetric=True, **kwargs, reduction='none')
    
    def compute_metric(self, gt, pred):
        return self.sd(gt, pred)

class LPIPSMetric:
    def __init__(self, net='alex'):
        sys.stdout = open(os.devnull, 'w')
        self.loss_fn = lpips.LPIPS(net=net)
        sys.stdout = sys.__stdout__  # Restore stdout

    def __call__(self, img1, img2, mask=None):
        assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"
        if mask is not None:
            assert mask.shape == img1.shape, f"Invalid mask shape: {mask.shape} {img1.shape}"
        
        #LPIPS takes images as input with range [-1,1]
        img1 = img1 * 2 - 1
        img2 = img2 * 2 - 1

        self.loss_fn = self.loss_fn.to(img1.device)

        if len(img1.shape) == 5:
            return self._compute_3d_lpips(img1, img2, mask)
        else:
            return self.loss_fn(img1, img2).view(img1.shape[0])

    def _compute_3d_lpips(self, img1, img2, mask):
        lpips_list = []
        for i in range(img1.shape[2]):
            if mask is not None:
                non_zero_mask = (mask.sum(dim=(0, 1, 3, 4)) > 0)
                if not non_zero_mask[i]:
                    continue

            lpips_list.append(self.loss_fn(img1[:, :, i, :, :], img2[:, :, i, :, :]))
        
        return torch.stack(lpips_list)if lpips_list else torch.tensor(0.0, device=img1.device)

class MAEMetric:
    def __init__(self, eps=1e-8):
        self.name = "MAE"
        self.eps = eps

    def __call__(self, img1, img2, mask=None):
        assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"
        if mask is not None:
            assert mask.shape == img1.shape, f"Invalid mask shape: {mask.shape} {img1.shape}"

        n, *_ = img1.shape
        img1 = img1.reshape(n, -1)
        img2 = img2.reshape(n, -1)
        if mask is not None:
            mask = mask.reshape(n, -1)

        diff = torch.abs(img1 - img2)

        if mask is not None:
            mask = (mask > 0).float()
            mae_values = (diff * mask).sum(dim=[-1]) / (mask.sum(dim=[-1]) + self.eps)
        else:
            mae_values = diff.mean(dim=[-1])

        return mae_values

class MSEMetric:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, img1, img2, mask=None):
        assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"
        if mask is not None:
            assert mask.shape == img1.shape, f"Invalid mask shape: {mask.shape} {img1.shape}"

        n, *_ = img1.shape
        img1 = img1.reshape(n, -1)
        img2 = img2.reshape(n, -1)
        if mask is not None:
            mask = mask.reshape(n, -1)

        squared_diff = (img1 - img2)**2

        if mask is not None:
            mask = (mask > 0).float()
            mse_values = (squared_diff * mask).sum(dim=[-1]) / (mask.sum(dim=[-1]) + self.eps)
        else:
            mse_values = squared_diff.mean(dim=[-1])

        return mse_values



class PSNRMetric:
    def __init__(self, three_d=False):
        self.name = "PSNR" if not three_d else "PSNR3D"
        self.three_d = three_d
        self.mse_loss = MSEMetric()

    def __call__(self, img1, img2, mask=None):
        assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"
        if mask is not None:
            assert mask.shape == img1.shape, f"Invalid mask shape: {mask.shape} {img1.shape}"
        v_max = 1.

        if len(img1.shape) == 5 and not self.three_d:
            psnrs = []
            for t in range(img1.shape[2]):
                mses = self.mse_loss(img1[:,:,t,:,:], img2[:,:,t,:,:], mask[:,:,t,:,:] if mask is not None else None)
                psnrs.append(20 * torch.log10(v_max / torch.sqrt(mses)))
            
            return torch.stack(psnrs).mean(dim=0)
        else:
            mses = self.mse_loss(img1, img2, mask=mask)
            psnr_values = 20 * torch.log10(v_max / torch.sqrt(mses))
            return psnr_values

class SSIMMetric:
    def __init__(self, *args, **kwargs):
        self.ssim = SSIM(*args, **kwargs)

    def __call__(self, img1, img2, mask=None):
        assert img1.shape == img2.shape, f"Shape mismatch: {img1.shape} vs {img2.shape}"
        res = self.ssim(img1, img2)
        return res.view(img1.shape[0])
    
class AccuracyMetric:
    """
    Computes top-1 token prediction accuracy between predicted logits and target token IDs.
    Expects:
        pred_logits: torch.Tensor of shape [B, N, vocab_size]
        target_ids:  torch.Tensor of shape [B, N]
    Returns:
        accuracy per sample: torch.Tensor of shape [B]
    """
    def __init__(self, topk=(1,)):
        self.topk = topk

    def __call__(self, logits, target_ids):
        """
        logits: [B, N, vocab_size] — model's predicted token scores
        target_ids: [B, N] — proxy tokenizer target IDs
        """
        B, N, V = logits.shape
        logits_flat = logits.view(B * N, V)    # [B*N, V]
        target_flat = target_ids.view(B * N)   # [B*N]

        # For each k in topk, compute the accuracy
        res = {}
        maxk = max(self.topk)
        
        # Get top maxk predictions (indices) for each token position
        # topk returns (values, indices), we only need indices here
        _, pred_topk = torch.topk(logits_flat, maxk, dim=-1, largest=True, sorted=True)  # [B*N, maxk]

        # Compare target with top-k predictions: for each position check if target_flat is in pred_topk row
        # Expand target_flat to shape [B*N, maxk] for comparison
        target_expanded = target_flat.unsqueeze(1).expand_as(pred_topk)

        correct = (pred_topk == target_expanded)  # [B*N, maxk], bool tensor

        for k in self.topk:
            # Check if correct in top-k (any True in first k columns)
            correct_k = correct[:, :k].any(dim=1).float()
            acc_k = correct_k.mean()
            res[f'top_{k}_accuracy'] = acc_k

        # If only one top-k requested, return just that value, else return dict
        if len(self.topk) == 1:
            return res[f'top_{self.topk[0]}_accuracy']
        return res