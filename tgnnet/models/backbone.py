"""Res2Net backbone used as the image stream encoder of TG-Net.

Ported from the official Res2Net release (Gao et al., "Res2Net: A New
Multi-scale Backbone Architecture", TPAMI 2021) with the following changes:

* the pretrained checkpoints are loaded by explicit path instead of
  ``model_zoo.load_url(model.urls[...])``, which referenced a module-level
  ``urls`` attribute that does not exist and raised ``AttributeError``;
* weight initialisation is applied through a public helper.
"""

import math

import torch
import torch.nn as nn

__all__ = [
    "Bottle2neck",
    "Res2Net",
    "res2net50_v1b_26w_4s",
    "res2net101_v1b_26w_4s",
    "res2net152_v1b_26w_4s",
]


class Bottle2neck(nn.Module):
    """Res2Net bottleneck with a hierarchical residual-like connection."""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 baseWidth=26, scale=4, stype="normal"):
        super().__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        self.nums = 1 if scale == 1 else scale - 1
        if stype == "stage":
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

        convs, bns = [], []
        for _ in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride,
                                   padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == "stage":
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.relu(self.bns[i](self.convs[i](sp)))
            out = sp if i == 0 else torch.cat((out, sp), 1)

        if self.scale != 1 and self.stype == "normal":
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == "stage":
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)

        return self.relu(out + residual)


class Res2Net(nn.Module):
    """Res2Net with an ImageNet-style stem kept at stride 4 after conv1.

    The first two convolutions are strided / non-strided as in the original
    v1b variant, so a 352x352 input yields the feature pyramid
    88 / 44 / 22 / 11 that TG-Net's reverse-attention branches expect.
    """

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        super().__init__()
        self.inplanes = 64
        self.baseWidth = baseWidth
        self.scale = scale

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [
            block(self.inplanes, planes, stride, downsample=downsample,
                  stype="stage", baseWidth=self.baseWidth, scale=self.scale)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def _load_pretrained(model, checkpoint, strict=True):
    """Load an ImageNet checkpoint, tolerating the ``module.`` prefix left by
    DataParallel training and transparently dropping the classification head."""
    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("state_dict", state) if isinstance(state, dict) else state
    state = {k.replace("module.", "", 1): v for k, v in state.items()}

    if not strict:
        model_dict = model.state_dict()
        state = {k: v for k, v in state.items()
                 if k in model_dict and v.shape == model_dict[k].shape}

    model.load_state_dict(state, strict=strict)
    return model


def res2net50_v1b_26w_4s(pretrained=False, checkpoint=None, **kwargs):
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        if checkpoint is None:
            raise ValueError(
                "pretrained=True requires --pretrained_path / MODEL.PRETRAINED_PATH "
                "pointing to a res2net50_v1b_26w_4s checkpoint."
            )
        _load_pretrained(model, checkpoint, strict=False)
    return model


def res2net101_v1b_26w_4s(pretrained=False, checkpoint=None, **kwargs):
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained and checkpoint is not None:
        _load_pretrained(model, checkpoint, strict=False)
    return model


def res2net152_v1b_26w_4s(pretrained=False, checkpoint=None, **kwargs):
    model = Res2Net(Bottle2neck, [3, 8, 36, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained and checkpoint is not None:
        _load_pretrained(model, checkpoint, strict=False)
    return model
