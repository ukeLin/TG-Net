"""The assembled TG-Net.

Pipeline (Fig. 2 of the paper):

    image ---------------------> Res2Net encoder -----------+
                                  x2 (1/4)  x3 (1/8)  x4 (1/16)
                                     |         |         |
                                    RFB       RFB       RFB
                                     |         |         |
                                     +----- MLF --------+--> global map g
                                                              |
    clinical prompt -> TA block ----------------------+        |
                                                       \\       |
                                              MFB fusion -------> g'
                                                                  |
                                            MSRA decoder (3 branches)
                                                                  |
                                    Sup-2 (1/4), Sup-3 (1/8), Sup-4 (1/16), Sup-5 (= g')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import res2net50_v1b_26w_4s
from .blocks import RFB_modified
from .fusion import MFBFusion
from .mlf import MLF
from .reverse_attention import ReverseAttentionBranch
from .text_attention import TextAttentionBlock

__all__ = ["TGNet", "build_model"]


class TGNet(nn.Module):
    """Text-guided skin lesion segmentation network.

    Args:
        channel: common width of the RFB / MLF / MFB path.
        n_class: output channels, 1 for binary segmentation.
        prompt_dim: number of categorical fields in the clinical prompt.
        text_hidden: hidden widths of the text attention trunk.
        mm_dim, factor: MFB operator hyper-parameters.
        frozen_backbone: if True, the Res2Net encoder is frozen and only the
            prompt-conditioned decoder is trained (useful for fine-tuning on
            small sets such as PH2 with 200 images).
        pretrained_backbone: path to a res2net50_v1b_26w_4s checkpoint.
    """

    def __init__(self, channel=32, n_class=1, prompt_dim=5, text_hidden=(14, 32),
                 mm_dim=1200, factor=2, frozen_backbone=False,
                 pretrained_backbone=None, **mfb_kwargs):
        super().__init__()
        self.channel = channel
        self.n_class = n_class
        self.prompt_dim = prompt_dim

        # ---- image stream ------------------------------------------------- #
        self.resnet = res2net50_v1b_26w_4s(
            pretrained=bool(pretrained_backbone),
            checkpoint=pretrained_backbone,
        )

        # ---- receptive field blocks --------------------------------------- #
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)

        # ---- multi-level fusion ------------------------------------------- #
        self.mlf = MLF(channel, n_class)

        # ---- text stream -------------------------------------------------- #
        self.text_attention = TextAttentionBlock(
            input_dim=prompt_dim, dimensions=text_hidden, **{
                k: v for k, v in mfb_kwargs.items()
                if k in ("activation", "dropout")
            }
        )

        # ---- text-image fusion -------------------------------------------- #
        self.fusion = MFBFusion(text_dim=self.text_attention.output_dim,
                                mm_dim=mm_dim, factor=factor, **mfb_kwargs)
        # ---- multi-scale reverse attention decoder ------------------------ #
        # Each branch resizes the running prediction onto its own encoder grid,
        # then upsamples its context by 16x / 16x / 8x to reach the 1/2
        # intermediate resolution; ``test.py`` upsamples Sup-2 (the headline
        # output) to the original image size for the metrics.
        self.ra4 = ReverseAttentionBranch(2048, 256, n_class,
                                          kernel_size=5, n_conv=4,
                                          output_scale=16)
        self.ra3 = ReverseAttentionBranch(1024, 64, n_class,
                                          kernel_size=3, n_conv=3,
                                          output_scale=16)
        self.ra2 = ReverseAttentionBranch(512, 64, n_class,
                                          kernel_size=3, n_conv=3,
                                          output_scale=8)

        if frozen_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

        self.n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---------------------------------------------------------------------- #
    def forward(self, x, prompt, output_size=None):
        """Args:
            x: (B, 3, S, S) dermoscopy images.
            prompt: (B, prompt_dim) categorical clinical records.
            output_size: optional (H, W) to resize every side output to. When
                omitted the outputs keep their native resolutions, and Sup-2 /
                Sup-3 come out at half the input size. The training loop passes
                the ground-truth size so that all four maps can be scored
                against the mask directly.
        Returns:
            (sup1, sup2, sup3, sup4) logits, ordered from the coarsest
            supervision (the prompt-modulated global map) to the finest RA
            branch.
        """
        # ---- encoder ------------------------------------------------------ #
        x = self.resnet.maxpool(self.resnet.relu(self.resnet.bn1(self.resnet.conv1(x))))
        x1 = self.resnet.layer1(x)   # (B, 256, S/4, S/4)
        x2 = self.resnet.layer2(x1)  # (B, 512, S/8, S/8)
        x3 = self.resnet.layer3(x2)  # (B, 1024, S/16, S/16)
        x4 = self.resnet.layer4(x3)  # (B, 2048, S/32, S/32)

        x2_rfb = self.rfb2_1(x2)
        x3_rfb = self.rfb3_1(x3)
        x4_rfb = self.rfb4_1(x4)

        # ---- multi-level fusion -> global map ----------------------------- #
        mlf = self.mlf(x4_rfb, x3_rfb, x2_rfb)

        # ---- text attention + fusion -------------------------------------- #
        text_feat = self.text_attention(prompt)
        mlf = self.fusion(mlf, text_feat)

        # Sup-1: the prompt-modulated global map, upsampled to the input
        # resolution (44x44 -> 352x352 at trainsize=352).
        sup1 = F.interpolate(mlf, scale_factor=8, mode="bilinear",
                             align_corners=False)

        # ---- reverse attention decoder ------------------------------------ #
        context4, sup2 = self.ra4(x4, mlf)
        context3, sup3 = self.ra3(x3, context4)
        _, sup4 = self.ra2(x2, context3)

        outputs = [sup1, sup2, sup3, sup4]
        if output_size is not None:
            target = tuple(output_size) if not torch.is_tensor(output_size) \
                else tuple(output_size.shape[-2:])
            outputs = [
                o if o.shape[-2:] == target else
                F.interpolate(o, size=target, mode="bilinear", align_corners=False)
                for o in outputs
            ]
        return tuple(outputs)


def build_model(cfg, prompt_dim, pretrained_backbone=None):
    """Instantiate TG-Net from a :class:`tgnnet.config.Config`.

    The fusion projections are registered for the spatial size implied by
    ``cfg.train.trainsize`` (352 -> 44, 384 -> 48, 416 -> 52, ...), so training
    at a non-default resolution needs no manual intervention.
    """
    m = cfg.model
    trainsize = int(cfg.train.get("trainsize", 352))
    spatial = (trainsize // 32) * 4

    model = TGNet(
        channel=m.get("channel", 32),
        n_class=m.get("n_class", 1),
        prompt_dim=prompt_dim,
        text_hidden=tuple(m.get("text_hidden", (14, 32))),
        mm_dim=m.get("mm_dim", 1200),
        factor=m.get("factor", 2),
        frozen_backbone=m.get("frozen_backbone", False),
        pretrained_backbone=pretrained_backbone,
        activ_input=m.get("activ_input", "relu"),
        activ_output=m.get("activ_output", "relu"),
        normalize=m.get("normalize", True),
        dropout_input=m.get("dropout_input", 0.0),
        dropout_pre_norm=m.get("dropout_pre_norm", 0.0),
        dropout_output=m.get("dropout_output", 0.0),
    )
    model.fusion.set_spatial_sizes([spatial])
    return model
