"""Loss functions used by TG-Net."""

import torch
import torch.nn.functional as F


def structure_loss(pred, mask):
    """Weighted BCE + weighted IoU, as used in the PraNet / MSNet family.

    Args:
        pred: logits of shape (B, 1, H, W).
        mask: binary ground-truth of shape (B, 1, H, W).

    The per-pixel weight ``weit`` emphasises pixels near the lesion boundary,
    which is what makes this loss suitable for the eroded / blurred borders
    that dominate dermoscopy images.
    """
    weit = 1 + 5 * torch.abs(
        F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask
    )
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    prob = torch.sigmoid(pred)
    inter = ((prob * mask) * weit).sum(dim=(2, 3))
    union = ((prob + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def multi_branch_structure_loss(predictions, mask, weights=None):
    """Sum the structure loss over the four decoder branches.

    Args:
        predictions: iterable of logits, ordered [sup1, sup2, sup3, sup4]
            (i.e. outer-most to inner-most branch).
        mask: binary ground-truth of shape (B, 1, H, W).
        weights: optional per-branch weights, default all ones.
    """
    if weights is None:
        weights = [1.0] * len(predictions)
    if len(weights) != len(predictions):
        raise ValueError("weights must have the same length as predictions")

    total = None
    per_branch = []
    for weight, pred in zip(weights, predictions):
        branch_loss = structure_loss(pred, mask)
        per_branch.append(branch_loss)
        total = branch_loss * weight if total is None else total + branch_loss * weight
    return total, per_branch
