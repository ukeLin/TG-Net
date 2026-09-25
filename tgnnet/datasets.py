"""Dataset definitions for the three benchmarks used in the paper.

Layout expected on disk (produced by ``preprocess/prepare_dataset.py``)::

    data/<DATASET>/
    ├── images/      image files (jpg / png / bmp)
    ├── masks/       binary ground-truth with matching file names
    └── *.csv        one CSV per case holding the clinical prompt fields

Design notes
------------
The released code defined three near-identical ``data.Dataset`` subclasses
(``ISIC_2017_csvDataset``, ``ph2_csvDataset``, ``Ham_csvDataset``) whose only
differences were the column list and the ``BatchNorm1d`` feature count. They
are merged here into a single :class:`PromptDataset` driven by a per-dataset
field specification, so adding a dataset means adding one :class:`DatasetSpec`.

Prompt dimensionality is dataset-dependent and must match the number of fields
declared in the spec, because the categorical indices are produced by fitting a
``LabelEncoder`` inside every individual CSV file (see
``tgnnet/models/text_attention.py`` for why).
"""

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from sklearn.preprocessing import LabelEncoder

from .utils import binary_loader, rgb_loader

__all__ = [
    "DatasetSpec",
    "DATASET_SPECS",
    "PromptDataset",
    "InferenceDataset",
    "PromptProvider",
    "build_dataloader",
    "build_inference_loader",
    "resolve_spec",
]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


class DatasetSpec:
    """Field layout of one benchmark's clinical prompt CSV."""

    def __init__(self, name, fields, image_exts=IMAGE_EXTS, note=""):
        self.name = name
        self.fields = list(fields)
        self.image_exts = tuple(image_exts)
        self.note = note

    @property
    def prompt_dim(self):
        return len(self.fields)

    def __repr__(self):
        return f"DatasetSpec({self.name!r}, prompt_dim={self.prompt_dim})"


DATASET_SPECS = {
    # ISIC 2017 ships two clinical columns -> 2-dimensional prompt.
    "isic2017": DatasetSpec(
        "isic2017",
        ["age_approximate", "sex"],
        note="ISIC 2017 training set; columns: age_approximate, sex",
    ),
    # PH2 carries the ABCD-rule annotations -> 16-dimensional prompt.
    "ph2": DatasetSpec(
        "ph2",
        [
            "Histological Diagnosis", "Common Nevus", "Atypical Nevus",
            "Melanoma", "Asymmetry", "Pigment Network", "Dots/Globules",
            "Streaks", "Regression Areas", "Blue-Whitish Veil", "White",
            "Red", "Light-Brown", "Dark-Brown", "Blue-Gray", "Black",
        ],
        note="PH2; 16 ABCD-rule / diagnosis fields",
    ),
    # HAM10000 metadata -> 5-dimensional prompt.
    "ham10000": DatasetSpec(
        "ham10000",
        ["dx", "dx_type", "age", "sex", "localization"],
        note="HAM10000 metadata; columns: dx, dx_type, age, sex, localization",
    ),
}


def resolve_spec(dataset):
    """Look up a spec by name, tolerating common aliases."""
    key = str(dataset).lower().replace("-", "").replace("_", "")
    aliases = {
        "isic2017": "isic2017",
        "isic17": "isic2017",
        "ph2": "ph2",
        "ham10000": "ham10000",
        "ham10k": "ham10000",
        "ham": "ham10000",
    }
    if key not in aliases:
        raise KeyError(
            f"unknown dataset {dataset!r}; available: {sorted(DATASET_SPECS)}"
        )
    return DATASET_SPECS[aliases[key]]


# --------------------------------------------------------------------------- #
# Prompt stream
# --------------------------------------------------------------------------- #
class PromptProvider:
    """Loads clinical prompts from the ``*.csv`` files of one split.

    Every CSV contributes one sample, and the sample order is fixed by the
    sorted file listing (``sorted(os.listdir)``) so that it can be aligned with
    the sorted image listing. ``BatchNorm1d`` is kept (rather than a plain
    standardisation layer) to stay numerically identical to the released code.
    """

    def __init__(self, csv_root, spec, feature_dim=None):
        self.csv_root = csv_root
        self.spec = spec
        self.feature_dim = feature_dim if feature_dim is not None else spec.prompt_dim
        self.tensor = self._load(csv_root)

    def _load(self, folder):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"prompt CSV directory not found: {folder}")

        frames = []
        for file_name in sorted(os.listdir(folder)):
            if not file_name.lower().endswith(".csv"):
                continue
            frame = pd.read_csv(os.path.join(folder, file_name))
            frame = frame[self.spec.fields].apply(
                lambda col: LabelEncoder().fit_transform(col)
            )
            frames.append(np.asarray(frame, dtype=np.float32))

        if not frames:
            raise FileNotFoundError(f"no .csv files found under {folder}")

        tensor = torch.from_numpy(np.concatenate(frames, axis=0))

        if tensor.size(1) != self.feature_dim:
            raise ValueError(
                f"{self.spec.name}: expected {self.feature_dim} prompt fields but "
                f"the CSV files provide {tensor.size(1)}. Check the `fields` list "
                f"of DatasetSpec and the dataset preparation script."
            )

        bn = nn.BatchNorm1d(self.feature_dim)
        return bn(tensor).detach()

    def __len__(self):
        return self.tensor.size(0)

    def __getitem__(self, index):
        return self.tensor[index]


# --------------------------------------------------------------------------- #
# Training / evaluation dataset
# --------------------------------------------------------------------------- #
class PromptDataset(data.Dataset):
    """Paired (image, mask, prompt) dataset.

    Args:
        image_root: folder holding the images.
        mask_root: folder holding the binary masks.
        trainsize: images are resized to ``trainsize x trainsize``.
        spec: a :class:`DatasetSpec`.
        csv_root: folder holding the per-case prompt CSVs. When ``None`` the
            prompt stream is disabled and ``__getitem__`` returns ``(image, mask)``.
        split: used only for error messages and the prompt/ordinal check.
    """

    def __init__(self, image_root, mask_root, trainsize, spec,
                 csv_root=None, split="train"):
        self.image_root = image_root
        self.mask_root = mask_root
        self.trainsize = trainsize
        self.spec = spec
        self.split = split

        self.images = self._list(image_root, spec.image_exts, "images")
        self.masks = self._list(mask_root, (".png", ".bmp", ".jpg", ".jpeg"), "masks")
        self._filter_mismatched()

        self.size = len(self.images)
        if self.size == 0:
            raise RuntimeError(f"{split}: no image/mask pairs found in {image_root}")

        self.prompt = None
        if csv_root is not None:
            self.prompt = PromptProvider(csv_root, spec)
            if len(self.prompt) != self.size:
                raise RuntimeError(
                    f"{split}: {len(self.prompt)} prompt CSV entries but {self.size} "
                    f"images -- the two streams must stay index-aligned."
                )

        self.img_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
        ])

    # -- helpers ----------------------------------------------------------- #
    @staticmethod
    def _list(folder, exts, what):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"{what} directory not found: {folder}")
        return [os.path.join(folder, n) for n in sorted(os.listdir(folder))
                if n.lower().endswith(exts)]

    def _filter_mismatched(self):
        """Keep only pairs whose stems match, reporting the dropped files."""
        from .utils import stem

        mask_by_stem = {}
        for path in self.masks:
            mask_by_stem.setdefault(stem(path), path)

        pairs = [(img, mask_by_stem[stem(img)])
                 for img in self.images if stem(img) in mask_by_stem]
        dropped = len(self.images) - len(pairs)
        if dropped:
            print(f"[{self.split}] dropped {dropped} image(s) without a matching mask")

        if pairs:
            self.images = [p[0] for p in pairs]
            self.masks = [p[1] for p in pairs]
        else:
            self.images, self.masks = [], []

    # -- dataset protocol --------------------------------------------------- #
    def __getitem__(self, index):
        image = self.img_transform(rgb_loader(self.images[index]))
        mask = self.gt_transform(binary_loader(self.masks[index]))
        if self.prompt is None:
            return image, mask
        return image, mask, self.prompt[index]

    def __len__(self):
        return self.size


class InferenceDataset(data.Dataset):
    """Images only, used by ``test.py`` to write predictions."""

    def __init__(self, image_root, testsize, spec, csv_root=None):
        self.testsize = testsize
        self.spec = spec
        self.images = PromptDataset._list(image_root, spec.image_exts, "images")
        self.size = len(self.images)
        self.transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.prompt = PromptProvider(csv_root, spec) if csv_root else None

    def __getitem__(self, index):
        path = self.images[index]
        with Image.open(path) as handle:
            original_size = handle.size  # (width, height)
        image = self.transform(rgb_loader(path))
        prompt = self.prompt[index] if self.prompt is not None else torch.zeros(0)
        return image, prompt, os.path.basename(path), original_size

    def __len__(self):
        return self.size


# --------------------------------------------------------------------------- #
# Factory helpers
# --------------------------------------------------------------------------- #
def _make_loader(dataset, batchsize, shuffle, num_workers, pin_memory):
    return data.DataLoader(dataset=dataset,
                           batch_size=batchsize,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=pin_memory,
                           drop_last=False)


def build_dataloader(image_root, mask_root, trainsize, spec, csv_root=None,
                     batchsize=3, shuffle=True, num_workers=4,
                     pin_memory=True, split="train"):
    dataset = PromptDataset(image_root, mask_root, trainsize, spec,
                            csv_root=csv_root, split=split)
    loader = _make_loader(dataset, batchsize, shuffle, num_workers, pin_memory)
    return loader, dataset


def build_inference_loader(image_root, testsize, spec, csv_root=None,
                           batchsize=1, num_workers=2):
    dataset = InferenceDataset(image_root, testsize, spec, csv_root=csv_root)
    loader = _make_loader(dataset, batchsize, False, num_workers, False)
    return loader, dataset
