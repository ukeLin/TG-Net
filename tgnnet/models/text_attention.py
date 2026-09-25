"""Text attention (TA) block.

Paper reference: "We proposed a text attention block, which extracts features
from textual prompts (such as lesion type, diagnostic methods, age, gender, and
lesion location) to guide skin lesion segmentation."

Implementation notes
--------------------
Each clinical record is a vector of *categorical* fields (for HAM10000: dx,
dx_type, age, sex, localization; for ISIC 2017: age_approximate, sex; for PH2:
histological diagnosis plus the ABCD-rule attributes). Two design choices in
the reference implementation are load-bearing and therefore preserved:

1. Every candidate ``csv`` folder is encoded with ``LabelEncoder().fit_transform``
   *independently*, so category indices restart at 0 per file. The prompt
   dimension is therefore a function of the CSV layout, not of a global
   vocabulary (see ``tgnnet/datasets.py``).
2. ``DataLoader(shuffle=True)`` is intentionally left on: the image and prompt
   streams are zipped positionally, and shuffling both with the same seed keeps
   them aligned while providing the regularisation the paper relies on.
"""

import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TextAttentionBlock", "MLP"]


class TextAttentionBlock(nn.Module):
    """Embed a clinical prompt vector into a fixed-width condition vector.

    Args:
        input_dim: number of categorical prompt fields in the CSV.
        dimensions: hidden widths of the MLP trunk.
        activation: name of a ``torch.nn.functional`` activation.
        dropout: dropout probability between hidden layers.
    """

    def __init__(self, input_dim=5, dimensions=(14, 32),
                 activation="relu", dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.dimensions = list(dimensions)
        self.activation = activation
        self.dropout = dropout

        self.linears = nn.ModuleList([nn.Linear(input_dim, self.dimensions[0])])
        for din, dout in zip(self.dimensions[:-1], self.dimensions[1:]):
            self.linears.append(nn.Linear(din, dout))
        self.output_dim = self.dimensions[-1]

    def forward(self, x):
        x = x.float()
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < len(self.linears) - 1:
                x = getattr(F, self.activation)(x)
                if self.dropout > 0:
                    x = F.dropout(x, self.dropout, training=self.training)
        return x


# Backwards-compatible alias: the released code and the paper's response to
# review both refer to this component as "MLP" inside the text stream.
MLP = TextAttentionBlock
