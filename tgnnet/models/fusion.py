"""Text-image fusion branch (MFB operator).

After the MLF module produces a global map, the prompt embedding is projected
to the same spatial footprint and combined with the flattened global map
through a factorised outer product followed by a signed square-root
composition. This is the "hierarchical guidance" step that injects the prompt
into the image stream.

The projection target depends on the spatial size of the MLF output, so the
module keeps one linear layer per registered resolution. The defaults cover the
three sizes that the paper's ``trainsize`` values produce::

    trainsize 352 -> 44x44 global map
    trainsize 384 -> 48x48 global map
    trainsize 416 -> 52x52 global map

Registering further sizes is a one-liner via :meth:`MFBFusion.set_spatial_sizes`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MFBFusion"]


class MFBFusion(nn.Module):
    """Hierarchical text guidance applied to the MLF global map.

    The prompt embedding is projected to the flattened spatial footprint of the
    global map, multiplied element-wise with the (also projected) global map in
    an ``mm_dim * factor`` space, pooled over ``factor`` groups and projected
    back. Because the projection depends on the global map's spatial size, one
    linear layer per supported resolution is kept (matching the reference
    implementation's ``linear1/2/3``); use :meth:`set_spatial_sizes` when
    training at a resolution the defaults do not cover.
    """

    def __init__(self, text_dim=32, mm_dim=1200, factor=2,
                 spatial_sizes=(44, 48, 52),
                 activ_input="relu", activ_output="relu",
                 normalize=True, dropout_input=0.0,
                 dropout_pre_norm=0.0, dropout_output=0.0):
        super().__init__()
        self.text_dim = text_dim
        self.mm_dim = mm_dim
        self.factor = factor
        self.activ_input = activ_input
        self.activ_output = activ_output
        self.normalize = normalize
        self.dropout_input = dropout_input
        self.dropout_pre_norm = dropout_pre_norm
        self.dropout_output = dropout_output

        self.linear0 = nn.Linear(text_dim, mm_dim * factor)
        self.linear_in = nn.ModuleDict()
        self.linear_out = nn.ModuleDict()
        for size in spatial_sizes:
            self.add_spatial_size(size)

    # -- construction ------------------------------------------------------ #
    def add_spatial_size(self, size):
        """Register input/output projections for one spatial resolution."""
        key = str(size)
        if key not in self.linear_in:
            self.linear_in[key] = nn.Linear(size * size, self.mm_dim * self.factor)
            self.linear_out[key] = nn.Linear(self.mm_dim, size * size)

    @property
    def spatial_sizes(self):
        return sorted(int(k) for k in self.linear_in.keys())

    def set_spatial_sizes(self, sizes):
        """Ensure projections exist for every spatial size in ``sizes``."""
        for size in sizes:
            self.add_spatial_size(int(size))
        return self

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _dropout(x, p, training):
        return F.dropout(x, p=p, training=training) if p > 0 else x

    # -- forward ----------------------------------------------------------- #
    def forward(self, mlf, text_feat):
        """Args:
            mlf: global map of shape (B, 1, S, S) produced by the MLF module.
            text_feat: prompt embedding of shape (B, text_dim).
        Returns:
            the prompt-modulated global map, same shape as ``mlf``.
        """
        spatial = mlf.size(2)
        key = str(spatial)
        if key not in self.linear_in:
            raise ValueError(
                f"the fusion layer has no projection for a {spatial}x{spatial} "
                f"global map; known sizes are {self.spatial_sizes}. Call "
                f"set_spatial_sizes([{spatial}]) first, or use one of "
                f"--trainsize 352/384/416."
            )

        x0 = self.linear0(text_feat)
        x1 = self.linear_in[key](mlf.flatten(1))

        if self.activ_input:
            x0 = getattr(F, self.activ_input)(x0)
            x1 = getattr(F, self.activ_input)(x1)

        x0 = self._dropout(x0, self.dropout_input, self.training)
        x1 = self._dropout(x1, self.dropout_input, self.training)

        z = x0 * x1
        z = self._dropout(z, self.dropout_pre_norm, self.training)

        # Factorised pooling over the outer-product dimension.
        z = z.view(-1, int(z.size(1) / self.factor), self.factor).sum(2)

        if self.normalize:
            z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
            z = F.normalize(z, p=2, dim=0)

        z = self.linear_out[key](z)
        if self.activ_output:
            z = getattr(F, self.activ_output)(z)
        z = self._dropout(z, self.dropout_output, self.training)

        return mlf + z.reshape(mlf.size(0), 1, spatial, spatial)
