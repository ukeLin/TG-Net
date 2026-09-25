"""Configuration loading.

A run is fully described by one YAML file in ``configs/``. Command-line flags
in ``train.py`` / ``test.py`` override individual keys, so both are usable:

    python train.py --config configs/ham10000.yaml --epoch 150 --gpu 1
"""

import os
from copy import deepcopy

import yaml

__all__ = ["Config", "load_config", "REPO_ROOT"]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Config(dict):
    """A dict with attribute access and dotted ``get`` support."""

    def __getattr__(self, item):
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key, value):
        self[key] = value

    def get(self, key, default=None):
        value = super().get(key, default)
        return Config(value) if isinstance(value, dict) else value

    def resolve(self, *keys):
        """Return the first non-None value among ``keys`` (dotted paths allowed)."""
        for key in keys:
            node, found = self, True
            for part in key.split("."):
                if isinstance(node, dict) and part in node:
                    node = node[part]
                else:
                    found = False
                    break
            if found and node is not None:
                return node
        return None


def load_config(path):
    """Load a YAML config, filling in derived paths."""
    if not os.path.isabs(path):
        candidate = os.path.join(REPO_ROOT, path)
        path = candidate if os.path.exists(candidate) else path
    if not os.path.exists(path):
        raise FileNotFoundError(f"config not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        cfg = Config(yaml.safe_load(handle) or {})

    cfg["_config_path"] = os.path.abspath(path)
    cfg["_config_name"] = os.path.splitext(os.path.basename(path))[0]
    return cfg


def merge_overrides(cfg, overrides):
    """Apply ``{"section.key": value}`` overrides onto a config in place."""
    cfg = deepcopy(dict(cfg))
    for dotted, value in overrides.items():
        if value is None:
            continue
        parts, node = dotted.split("."), cfg
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return Config(cfg)


def dataset_paths(cfg):
    """Resolve the train/test image, mask and prompt paths of a config."""
    root = cfg.data.get("root", "data")
    if not os.path.isabs(root):
        root = os.path.join(REPO_ROOT, root)

    name = cfg.data.name
    train_dir = cfg.data.get("train_dir") or os.path.join(root, name)
    test_dir = cfg.data.get("test_dir") or train_dir
    csv_name = cfg.data.get("csv_subdir", "csv")

    def join(base, *parts):
        return os.path.join(base, *parts)

    return {
        "train_image_root": join(train_dir, "images") + os.sep,
        "train_mask_root": join(train_dir, "masks") + os.sep,
        # CSV folder: configurable because the released layout expected a
        # folder full of per-case CSVs, while consolidated datasets keep a
        # single metadata table at the split root.
        "train_csv_root": join(train_dir, csv_name) if csv_name else train_dir,
        "test_image_root": join(test_dir, "images") + os.sep,
        "test_mask_root": join(test_dir, "masks") + os.sep,
        "test_csv_root": join(test_dir, csv_name) if csv_name else test_dir,
        "save_root": join(root if os.path.isabs(root) else os.path.join(REPO_ROOT, root),
                          "..", cfg.train.get("save_dir", "Snapshots")),
    }
