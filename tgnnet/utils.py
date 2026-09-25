"""Small helpers shared by training, testing and data loading."""

import os
import random

import numpy as np
import torch
from PIL import Image


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed=42):
    """Make a run as reproducible as PyTorch reasonably allows."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Image I/O
# --------------------------------------------------------------------------- #
def rgb_loader(path):
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


def binary_loader(path):
    with open(path, "rb") as f:
        return Image.open(f).convert("L")


def list_images(folder, exts=(".jpg", ".jpeg", ".png", ".bmp")):
    """Return the sorted list of image paths inside ``folder``."""
    if not folder or not os.path.isdir(folder):
        return []
    names = [n for n in os.listdir(folder) if n.lower().endswith(exts)]
    return [os.path.join(folder, n) for n in sorted(names)]


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


# --------------------------------------------------------------------------- #
# Optimisation
# --------------------------------------------------------------------------- #
def clip_gradient(optimizer, grad_clip):
    """Clamp gradients of all parameters to [-grad_clip, grad_clip]."""
    if grad_clip is None or grad_clip <= 0:
        return
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, base_lr, epoch, decay_rate=0.05, decay_epoch=25):
    """Exponentially decay the learning rate every ``decay_epoch`` epochs."""
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = base_lr * decay
    return base_lr * decay


# --------------------------------------------------------------------------- #
# Metrics bookkeeping
# --------------------------------------------------------------------------- #
class AvgMeter:
    """Running average of a scalar, with a windowed view for logging."""

    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)
        self.losses.append(float(val))

    def show(self):
        if not self.losses:
            return 0.0
        recent = self.losses[max(len(self.losses) - self.num, 0):]
        return float(np.mean(recent))


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def normalize_map(arr):
    """Min-max normalise a 2D probability map to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)
