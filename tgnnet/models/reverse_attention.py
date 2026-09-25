"""Multi-scale reverse attention (MSRA) decoder.

Paper reference: "we integrated the multi-scale feature extraction with reverse
attention (RA), proposing multi-scale reverse attention (MSRA). MSRA improves
the ability of the model to extract image details of varying sizes, while
reversing the decoder's deep output enhances focus on the skin lesion boundary
efficiently."

Three branches operate on the encoder stages x4 (1/16), x3 (1/8) and x2 (1/4)
of the input. Each branch down-samples the prediction made so far to the
resolution of the encoder feature it attends to, inverts it, and uses the
result as a spatial weight on that feature:

    attn = 1 - sigmoid(downsample(prior))
    x    = attn * encoder_feat

so the branch is pushed to mine the residual, boundary-dominated evidence that
the coarser prediction has not yet explained. The branch output becomes the
prior of the next, shallower branch, giving the coarse-to-fine chain
Sup-4 -> Sup-3 -> Sup-2.

Resolution bookkeeping
----------------------
The prior has to be resized to the *encoder feature grid*, which is not a fixed
multiple of the prior size across branches. For a 352x352 input:

    branch   encoder feat   prior resolution   resized to   prediction
    RA4      x4   11x11     MLF    44x44        -> 11x11      -> 352x352
    RA3      x3   22x22     RA4   176x176       -> 22x22      -> 352x352
    RA2      x2   44x44     RA3    22x22        -> 44x44      -> 352x352

The resize is therefore done with an explicit target size rather than a scale
factor, which keeps the module correct for any input size divisible by 32.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import BasicConv2d

__all__ = ["ReverseAttentionBranch"]


class ReverseAttentionBranch(nn.Module):
    """One reverse-attention stage.

    Args:
        in_channels: channels of the encoder feature this branch attends to.
        mid_channels: internal width of the branch.
        n_class: number of output channels (1 for binary segmentation).
        kernel_size: spatial kernel of the internal convolutions; the deepest
            branch uses 5x5 in the reference implementation, the shallower
            ones 3x3.
        n_conv: number of intermediate conv layers (4 for the deepest branch,
            3 for the shallower ones).
        output_scale: factor by which the branch context is upsampled to the
            final prediction resolution (8 for RA2, 16 for RA3 and RA4 when
            ``trainsize`` is 352).
    """

    def __init__(self, in_channels, mid_channels, n_class=1,
                 kernel_size=5, n_conv=4, output_scale=8):
        super().__init__()
        padding = kernel_size // 2
        layers = [BasicConv2d(in_channels, mid_channels, kernel_size=1)]
        for _ in range(n_conv - 1):
            layers.append(BasicConv2d(mid_channels, mid_channels,
                                      kernel_size=kernel_size, padding=padding))
        self.body = nn.ModuleList(layers)
        self.out_conv = BasicConv2d(mid_channels, n_class, kernel_size=1)

        self.output_scale = output_scale

    def forward(self, encoder_feat, prior):
        """
        Args:
            encoder_feat: (B, C, H, W) encoder stage feature.
            prior: (B, 1, h, w) prediction from the previous branch.
        Returns:
            (context, prediction) where ``context`` is the pre-upsampling map
            passed to the next branch and ``prediction`` is the final mask.
        """
        if prior is None:
            raise ValueError("a prior map is required for every RA branch")

        # Resize the prior onto the encoder grid, preserving the prior's
        # resolution when it already matches.
        context = F.interpolate(prior, size=encoder_feat.shape[-2:],
                                mode="bilinear", align_corners=False)

        # Reverse (inverted) attention mask: suppress what has been found.
        attn = 1 - torch.sigmoid(context)
        x = attn * encoder_feat

        for i, layer in enumerate(self.body):
            x = layer(x)
            if i > 0:
                x = F.relu(x)

        context = self.out_conv(x) + context
        prediction = F.interpolate(context, scale_factor=self.output_scale,
                                   mode="bilinear", align_corners=False)
        return context, prediction

    def extra_repr(self):
        return f"output_scale={self.output_scale}"
