#!/usr/bin/env python
"""Convert a raw skin-lesion dataset into the layout TG-Net expects.

Usage
-----
    python preprocess/prepare_dataset.py --dataset ham10000 \
        --images /path/to/HAM10000_images \
        --masks  /path/to/HAM10000_segmentations \
        --metadata /path/to/HAM10000_metadata.csv \
        --out data/HAM10000

Result::

    data/HAM10000/
    ├── images/<case>.jpg      images resized to at most --max_side px
    ├── masks/<case>.png       binarised ground truth (0 / 255)
    └── csv/<case>.csv         prompt fields for that case

The ``csv`` folder holds one small CSV per case because that is what the data
loader reads (``PromptProvider`` iterates ``*.csv`` in sorted order and encodes
each file independently). If your dataset already ships per-case CSVs, pass
``--per_case_csv /path/to/folder`` and only the images/masks are produced.

Datasets this script knows about out of the box
-----------------------------------------------
* ``ham10000`` -- flat image folder + a metadata CSV keyed by ``image_id``
* ``isic2017`` -- image folder + a single CSV keyed by ``image``
* ``ph2``      -- image folder with ``<case>_<lesion>_<diagnosis>.bmp`` names
                  plus a metadata CSV/XLSX keyed by ``name``

For anything else, use ``--field-map`` to describe which columns form the
prompt; the accepted form is ``csv_column=label`` pairs, repeatable.
"""

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from tgnnet.datasets import DATASET_SPECS, resolve_spec  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_metadata(path):
    """Read a CSV or XLSX metadata table into a DataFrame."""
    if path is None:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    sep = "\t" if ext in (".tsv", ".txt") else ","
    return pd.read_csv(path, sep=sep)


def save_mask(source, destination, threshold=128):
    """Binarise a ground-truth mask and write it as an 8-bit PNG."""
    with Image.open(source) as handle:
        mask = handle.convert("L")
        array = np.asarray(mask)
    binary = (array > threshold).astype(np.uint8) * 255
    Image.fromarray(binary, mode="L").save(destination)


def maybe_resize(source, destination, max_side, quality=95):
    """Copy an image, optionally downscaling so the long side is ``max_side``."""
    with Image.open(source) as handle:
        image = handle.convert("RGB")
        if max_side and max(image.size) > max_side:
            scale = max_side / max(image.size)
            new_size = (int(round(image.width * scale)),
                        int(round(image.height * scale)))
            image = image.resize(new_size, Image.BILINEAR)
        ext = os.path.splitext(destination)[1].lower()
        if ext in (".jpg", ".jpeg"):
            image.save(destination, quality=quality)
        else:
            image.save(destination)


def index_by_stem(folder, exts=IMAGE_EXTS):
    """Map lowercase stem -> path for every image in ``folder``."""
    index = {}
    if folder is None or not os.path.isdir(folder):
        return index
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(exts):
            index.setdefault(os.path.splitext(name)[0].lower(),
                             os.path.join(folder, name))
    return index


def match_key(stem_name, metadata_keys):
    """Find the metadata row whose key prefixes the file stem (PH2 style)."""
    lowered = stem_name.lower()
    for key in metadata_keys():
        if lowered.startswith(str(key).lower()):
            return key
    return None


# --------------------------------------------------------------------------- #
# Per-dataset column resolution
# --------------------------------------------------------------------------- #
def resolve_case_ids(args, spec):
    """Return {case_id: (image_path, mask_path, metadata_row_index)} pairs."""
    images = index_by_stem(args.images)
    masks = index_by_stem(args.masks)
    metadata = load_metadata(args.metadata)

    if not images:
        raise SystemExit(f"no images found in {args.images}")

    if metadata is None:
        # No metadata: emit prompts of zeros so the pipeline still runs.
        return {stem: (path, masks.get(stem.lower()), None)
                for stem, path in images.items()}

    metadata = metadata.copy()
    id_column = args.id_column
    if id_column is None:
        for candidate in ("image_id", "image", "name", "case", "filename"):
            if candidate in metadata.columns:
                id_column = candidate
                break
    if id_column is None:
        raise SystemExit(
            "could not infer the id column; pass --id_column. "
            f"available columns: {list(metadata.columns)}"
        )

    key_to_row = {str(v): i for i, v in enumerate(metadata[id_column])}
    keys = list(key_to_row)

    cases = {}
    for stem, path in images.items():
        mask = masks.get(stem.lower())

        if stem in key_to_row:
            row = key_to_row[stem]
        else:
            hit = match_key(stem, lambda: keys)
            row = key_to_row.get(hit) if hit is not None else None

        cases[stem] = (path, mask, row)
    return cases


def build_prompt_frame(metadata, row_index, spec, field_map=None):
    """Extract the prompt fields for one case as a one-row DataFrame."""
    columns = []
    for field in spec.fields:
        source = (field_map or {}).get(field, field)
        if source not in metadata.columns:
            raise SystemExit(
                f"column {source!r} (needed for prompt field {field!r}) is "
                f"missing. Available: {list(metadata.columns)}. "
                f"Use --field-map {field}=<column> to remap."
            )
        columns.append(source)

    frame = metadata.iloc[[row_index]][columns].copy()
    frame.columns = spec.fields
    return frame


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare a skin-lesion dataset for TG-Net")
    parser.add_argument("--dataset", required=True,
                        help=f"one of {sorted(DATASET_SPECS)}")
    parser.add_argument("--images", required=True, help="folder with images")
    parser.add_argument("--masks", required=True, help="folder with masks")
    parser.add_argument("--metadata", default=None,
                        help="metadata CSV/XLSX holding the prompt fields")
    parser.add_argument("--per_case_csv", default=None,
                        help="folder of pre-existing per-case CSVs to copy")
    parser.add_argument("--out", required=True,
                        help="output root, e.g. data/HAM10000")
    parser.add_argument("--id_column", default=None,
                        help="metadata column matching the image file name")
    parser.add_argument("--field_map", action="append", default=None,
                        metavar="FIELD=COLUMN",
                        help="remap a prompt field to a metadata column")
    parser.add_argument("--max_side", type=int, default=None,
                        help="downscale images so the long side is at most this")
    parser.add_argument("--mask_threshold", type=int, default=128,
                        help="intensity above which a mask pixel is foreground")
    parser.add_argument("--overwrite", action="store_true",
                        help="clear the output directory first")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    spec = resolve_spec(args.dataset)

    field_map = {}
    for item in args.field_map or []:
        if "=" not in item:
            raise SystemExit(f"--field-map expects FIELD=COLUMN, got {item!r}")
        field, column = item.split("=", 1)
        field_map[field.strip()] = column.strip()

    out = args.out if os.path.isabs(args.out) else os.path.join(REPO_ROOT, args.out)
    if args.overwrite and os.path.isdir(out):
        shutil.rmtree(out)
    dirs = {name: os.path.join(out, name) for name in ("images", "masks", "csv")}
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    print(f"dataset : {spec}")
    print(f"fields  : {spec.fields}")
    print(f"output  : {out}")

    cases = resolve_case_ids(args, spec)
    total = len(cases)
    written, skipped = 0, []

    for stem, (image_path, mask_path, row_index) in sorted(cases.items()):
        if mask_path is None:
            skipped.append((stem, "no mask"))
            continue

        extension = os.path.splitext(image_path)[1].lower()
        if extension not in (".jpg", ".jpeg", ".png", ".bmp"):
            extension = ".jpg"
        image_out = os.path.join(dirs["images"], f"{stem}{extension}")
        mask_out = os.path.join(dirs["masks"], f"{stem}.png")
        csv_out = os.path.join(dirs["csv"], f"{stem}.csv")

        maybe_resize(image_path, image_out, args.max_side)
        save_mask(mask_path, mask_out, args.mask_threshold)

        if args.per_case_csv:
            source_csv = os.path.join(args.per_case_csv, f"{stem}.csv")
            if os.path.exists(source_csv):
                shutil.copyfile(source_csv, csv_out)
            else:
                skipped.append((stem, "per-case CSV not found"))
                continue
        elif row_index is not None:
            metadata = load_metadata(args.metadata)
            build_prompt_frame(metadata, row_index, spec, field_map).to_csv(
                csv_out, index=False)
        else:
            skipped.append((stem, "no metadata row"))
            continue

        written += 1
        if written % 500 == 0:
            print(f"  ... {written}/{total}")

    print(f"done. {written}/{total} cases written to {out}")
    if skipped:
        print(f"skipped {len(skipped)} case(s); first few:")
        for stem, reason in skipped[:10]:
            print(f"  - {stem}: {reason}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
