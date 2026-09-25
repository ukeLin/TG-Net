# Data

Datasets are **not** tracked in this repository. Download them from their
original sources and convert them with `preprocess/prepare_dataset.py`.

## Required layout

```
data/
└── <DATASET>/
    ├── images/          # one file per case: jpg / png / bmp
    ├── masks/           # binary ground truth, same stem as the image
    └── csv/             # one CSV per case holding the prompt fields
```

The `csv/` folder may be renamed via `data.csv_subdir` in the config; set it to
`""` if your split root holds the CSVs directly.

## Benchmark datasets

| Config | Dataset | Cases | Prompt fields (`DatasetSpec.fields`) | Source |
| --- | --- | --- | --- | --- |
| `configs/isic2017.yaml` | ISIC 2017 | 2000 | `age_approximate`, `sex` | <https://challenge.isic-archive.com/data/> |
| `configs/ham10000.yaml` | HAM10000 | 10015 | `dx`, `dx_type`, `age`, `sex`, `localization` | <https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T> |
| `configs/ph2.yaml` | PH2 | 200 | 16 ABCD-rule / diagnosis fields | <https://www.fc.up.pt/addi/ph2%20database.html> |

The prompt field lists are declared in `tgnnet/datasets.py` (`DATASET_SPECS`).
Adding a dataset is a matter of adding one `DatasetSpec` and one config.

## Converting a dataset

HAM10000 (flat image folder + a metadata table):

```bash
python preprocess/prepare_dataset.py \
    --dataset  ham10000 \
    --images   /path/to/HAM10000_images \
    --masks    /path/to/HAM10000_segmentations \
    --metadata /path/to/HAM10000_metadata.csv \
    --out      data/HAM10000 \
    --max_side 600
```

PH2 (file names encode the diagnosis; metadata lives in a spreadsheet):

```bash
python preprocess/prepare_dataset.py \
    --dataset  ph2 \
    --images   /path/to/PH2/images \
    --masks    /path/to/PH2/masks \
    --metadata /path/to/PH2/PH2_dataset.xlsx \
    --out      data/PH2
```

If the metadata column names differ from the prompt field names, remap them:

```bash
--field-map dx=diagnosis --field-map localization=lesion_location
```

## Splits

The paper reports results using the official splits of each benchmark. The
default configs point the test split at the same folder as the training split;
for a held-out evaluation, set `data.test_dir` explicitly:

```yaml
data:
  name: ham10000
  root: data
  train_dir: data/HAM10000_train
  test_dir:  data/HAM10000_test
  csv_subdir: csv
```

## Expected file sizes

Converted images keep their original aspect ratio. With `--max_side 600` the
HAM10000 subset is roughly 1.5 GB of images, which is enough for the 352x352
training size while keeping the repository-friendly footprint small.
