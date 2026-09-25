"""TG-Net: text-guided skin lesion segmentation.

Official implementation of

    TG-Net: Using text prompts for improved skin lesion segmentation
    Computers in Biology and Medicine, 2024

The package is organised by the modules described in the paper:

    models/backbone.py            Res2Net image encoder
    models/text_attention.py      Text attention (TA) / clinical prompt MLP
    models/blocks.py              RFB and the multi-level fusion (MLF) module
    models/fusion.py              MFB-style text-image fusion branch
    models/reverse_attention.py   Multi-scale reverse attention (MSRA) decoder
    models/tg_net.py              The assembled TG-Net
"""

from .models.tg_net import TGNet, build_model
from .datasets import PromptDataset, build_dataloader, resolve_spec
from .config import Config, load_config

__all__ = [
    "TGNet",
    "build_model",
    "PromptDataset",
    "build_dataloader",
    "resolve_spec",
    "Config",
    "load_config",
]

__version__ = "1.0.0"
