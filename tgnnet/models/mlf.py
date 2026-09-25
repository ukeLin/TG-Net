"""Multi-level fusion (MLF) module.

Paper reference: Section 3, "multi-level fusion (MLF) module to merge
higher-level features and generate a global feature map as guidance for
subsequent steps".

The module fuses the three RFB-projected encoder stages (x2, x3, x4) in a
top-down manner using gated multiplicative skip connections: a coarser level
is upsampled and used to weight the finer level, so that the coarse semantic
context modulates the fine spatial detail. The output is a single-channel
global map at the resolution of the 4x-downsampled stage.
"""

import torch
import torch.nn as nn

from .blocks import BasicConv2d

__all__ = ["MLF"]


class MLF(nn.Module):
    def __init__(self, channel, n_class=1):
        super().__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear",
                                    align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, n_class, 1)

    def forward(self, x1, x2, x3):
        """x1/x2/x3 are the RFB features at 1/4, 1/8 and 1/16 resolution."""
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = (self.conv_upsample2(self.upsample(self.upsample(x1)))
                * self.conv_upsample3(self.upsample(x2)) * x3)

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        return self.conv5(self.conv4(x3_2))
