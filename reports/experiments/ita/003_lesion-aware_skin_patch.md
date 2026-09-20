# Experiment 003 — Lesion-Aware Skin Patch

## Objective

Evaluate whether estimating skin tone from a lesion-free surrounding-skin patch reduces the influence of melanoma lesions on skin-tone estimation.

## Date

2026-09-20 20:59:50

## Input

- Dataset: ISIC 2020
- Subset: current test images
- Images processed: 6
- Image directory: `data/raw/isic2020/train`
- Original resolution: 512×512

## Change from Previous Experiment

A lesion-aware surrounding-skin crop was introduced after the multiscale hair-removal step. The lesion is detected using Otsu thresholding and border-connected components are excluded. A contiguous lesion-free skin patch surrounding the detected lesion is then selected for model inference. All other inference steps remain unchanged.

## Preprocessing

- BGR → RGB
- Multiscale blackhat hair detection
- Hair mask dilation
- Telea inpainting
- Otsu thresholding for lesion detection
- Removal of border-connected mask components
- Selection of a lesion-free surrounding-skin patch
- Resize selected patch to 224×224
- Convert pixel values to [0, 1]
- ImageNet normalization

## Model

- Architecture: EfficientNet-B4
- Task: Lab regression
- Output: L*, a*, b*
- Number of folds: 5
- Aggregation: mean L*, a*, b* across folds

## ITA Calculation

ITA = atan((L - 50) / b) × 180 / π

## Fitzpatrick Mapping

- ITA > 55 → I
- 41 < ITA ≤ 55 → II
- 28 < ITA ≤ 41 → III
- 19 < ITA ≤ 28 → IV
- 10 < ITA ≤ 19 → V
- ITA ≤ 10 → VI

## Results

| Image | L* | a* | b* | ITA | Fitzpatrick |
|---|---:|---:|---:|---:|---:|
| ISIC_9922133.jpg | 46.146 | 7.695 | 17.273 | -12.579 | 6 |
| ISIC_9922430.jpg | 53.471 | 6.807 | 16.634 | 11.788 | 5 |
| ISIC_9923018.jpg | 62.050 | 4.886 | 16.137 | 36.749 | 3 |
| ISIC_9886540.jpg | 57.161 | 10.463 | 15.358 | 24.996 | 4 |
| ISIC_9991451.jpg | 50.584 | 11.848 | 17.681 | 1.893 | 6 |
| ISIC_0068279.jpg | 51.532 | 10.024 | 17.058 | 5.133 | 6 |

## Observations

TODO

## Decision / Next Experiment

TODO
