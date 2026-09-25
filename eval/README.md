# Evaluation

MATLAB implementation of the seven segmentation metrics reported in the paper.
The code is the standard foreground-map evaluation toolbox used across the
medical segmentation literature, with the entry point adapted for TG-Net.

## Files

| File | Role |
| --- | --- |
| `main.m` | Entry point: threshold sweep, aggregation, results written to `../res/` |
| `StructureMeasure.m` | S-measure (Fan et al., ICCV'17) |
| `S_object.m`, `S_region.m` | Object-level / region-level terms of S-measure |
| `original_WFb.m` | Weighted F-measure (Margolin et al., CVPR'14) |
| `Enhancedmeasure.m` | E-measure (Fan et al., IJCAI'18) |
| `Fmeasure_calu.m` | Per-threshold Precision / Recall / Specificity / Dice / F1 / IoU |
| `CalMAE.m` | Mean absolute error (kept for reference; `main.m` computes MAE inline) |

## How to run

1. Generate prediction maps with the Python inference script:

   ```bash
   python test.py --config configs/ham10000.yaml \
                  --save_path results/HAM10000/outputs
   ```

   Each prediction is an 8-bit PNG named after its input image, exported at the
   original image resolution (so no resizing is needed at evaluation time).

2. Make sure the ground truth sits next to the predictions:

   ```
   results/
   └── HAM10000/
       ├── masks/     # ground truth, one PNG per case
       └── outputs/   # prediction maps from test.py
   ```

3. Set `Datasets` in `main.m` to `{'HAM10000'}` and run it from this folder:

   ```matlab
   >> cd eval
   >> main
   ```

## Output

* Per-dataset summary line printed to the command window and appended to
  `../res/<DATASET>_result.txt`:

  ```
  meanDic, meanIoU, wFm, Sm, meanEm, MAE, maxEm, maxDice, maxIoU,
  meanSen, maxSen, meanSpe, maxSpe
  ```

* A `.mat` file in `../res/<DATASET>-mat/` holding the full threshold curves
  (`column_Dic`, `column_IoU`, `column_Sen`, `column_Spe`, `column_E`) so plots
  can be redrawn without recomputing.

## Notes

* The threshold sweep is `1:-1/255:0` (256 steps). Metrics are first averaged
  per image across thresholds, then averaged across images.
* `meanDic` / `meanIoU` are the threshold-averaged values, so they may differ
  slightly from the "best-threshold" Dice quoted by papers that report a single
  operating point. Report whichever convention you use consistently.
* If a prediction's size does not match its ground-truth mask, `main.m` resizes
  the ground truth and prints a warning rather than failing.
