# TG-Net

**Using text prompts for improved skin lesion segmentation**

## Overview

TG-Net uses the clinical diagnostic record as a text prompt to guide dermoscopy
lesion segmentation: a **text attention (TA)** block embeds the prompt (lesion
type, diagnostic method, age, sex, lesion location), a **multi-level fusion
(MLF)** module merges higher-level features into a global guidance map, and a
**multi-scale reverse attention (MSRA)** decoder attends to the residual,
boundary-dominated evidence. Evaluated on ISIC 2017, HAM10000 and PH2.

## Usage

```bash
git clone https://github.com/ukeLin/TG-Net.git && cd TG-Net
pip install -r requirements.txt

# 1. convert a dataset to data/<DATASET>/{images,masks,csv}/
python preprocess/prepare_dataset.py --dataset ham10000 \
    --images /path/to/images --masks /path/to/masks \
    --metadata /path/to/metadata.csv --out data/HAM10000

# 2. train
python train.py --config configs/ham10000.yaml

# 3. predict (8-bit PNGs at original resolution)
python test.py --config configs/ham10000.yaml \
    --checkpoint Snapshots/ham10000/TGNet.pth \
    --save_path results/HAM10000/outputs
```

Then run `eval/main.m` in MATLAB for the metrics. Dataset downloads, prompt
field lists and conversion options: [`data/README.md`](data/README.md).
Prediction output and metric details: [`eval/README.md`](eval/README.md).

Python 3.9+, PyTorch 1.13+ (CUDA 11.7+ recommended).

## Layout

```
tgnnet/models/     backbone, TA, RFB, MLF, MFB fusion, MSRA, TG-Net
tgnnet/           datasets.py  losses.py  config.py  utils.py
configs/          one YAML per benchmark
eval/             MATLAB metrics
```

## Notes

* Training defaults (paper Sec. 4.1): Adam, lr 1e-4, batch 3, 352×352,
  100 epochs, decay ×0.05 every 25 epochs, clipping 0.5, multi-scale
  {0.75, 1.0, 1.25}. Snapshots go to `Snapshots/<experiment>/`.
* `test.py --branch` picks the output: `sup2` (default, deepest
  reverse-attention branch), `sup1` (prompt-modulated global map),
  `sup3`, `sup4`, or `mean`.
* PH2 has only 200 images — fine-tune from a HAM10000 checkpoint
  (`--pretrained_backbone Snapshots/ham10000/TGNet.pth --epoch 30`).

## Citation

```bibtex
@article{meng2024tg,
  title     = {TG-Net: Using text prompts for improved skin lesion segmentation},
  author    = {Meng, Xiangfu and Yu, Chunlin and Zhang, Zhichao and
               Zhang, Xiaoyan and Wang, Meng},
  journal   = {Computers in Biology and Medicine},
  volume    = {179},
  pages     = {108819},
  year      = {2024},
  publisher = {Elsevier}
}
```
