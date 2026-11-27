# Detecting Breast Cancer Lymph Node Metastases in Histopathology Patches

**Authors:** Eduard Bicu, Mihnea Zamfir. Data Science in Medicine (DSM), Faculty of Automatic Control and Computers, POLITEHNICA Bucharest.

Binary classification of 96×96 H&E patches from PatchCamelyon (PCam) as metastatic tumour or normal tissue.

## M2: Dataset and baseline results

The report is at `reports/m2/m2_report.pdf` (its source is `m2_report.tex`).

| Baseline | Test AUC [95% CI] | Accuracy | Recall | Specificity | F1 |
|---|---|---|---|---|---|
| Colour-only logistic regression (58 features) | 0.870 [0.867, 0.874] | 0.794 | 0.760 | 0.829 | 0.787 |
| ResNet-18, ImageNet-pretrained (epoch 4 of 10) | 0.928 [0.925, 0.931] | 0.844 | 0.732 | 0.955 | 0.824 |

## Layout

| Path | What it is |
|---|---|
| `eda.py` | Dataset audit: splits, slide and hospital provenance, duplicates, train/test leakage, stain statistics, figures, and the colour-only baseline. Writes to `reports/m2/`. |
| `train.py` | ResNet-18 training. The checkpoint is chosen by validation AUC and the test set is evaluated once. The M3 variants are flags (`--scratch`, `--label-frac`, `--aug`). |
| `pcam/` | Data loading (GPU-resident on CUDA, memmap otherwise), augmentation, metrics, model. |
| `colab_m2.ipynb` | Colab runner: downloads PCam with MD5 checks, trains, plots. |
| `reports/m2/` | Report, figures, `dataset_stats.json`, and the ResNet-18 run (`resnet18/`: config, per-epoch history, test metrics). |

## Reproduce

1. **Data:** put the PCam HDF5 files and the `*_meta.csv` files ([Zenodo 2546921](https://zenodo.org/records/2546921)) in `data/`. The Colab notebook downloads the HDF5 files itself.
2. **Dataset audit and colour baseline (CPU, about 5 min):** `python eda.py`
3. **ResNet-18 (GPU, about 10 min on a Colab T4):** `python train.py --name m2_baseline --amp`, or open `colab_m2.ipynb` in Colab and choose Run all.

Dependencies are listed in `requirements.txt`, with exact versions in `requirements.lock.txt`.
