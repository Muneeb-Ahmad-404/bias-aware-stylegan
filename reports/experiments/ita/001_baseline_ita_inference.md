# Experiment 001 — Baseline ITA Inference

## Objective
Establish baseline skin-tone estimates using the pretrained five-fold Lab regression models without additional preprocessing.

## Date
2026-09-20 19:35:14

## Input
- Dataset: ISIC 2020
- Subset: current test images
- Images processed: 6
- Image directory: `data/raw/isic2020/train`
- Original resolution: 512×512

## Preprocessing
- BGR → RGB
- Resize to 224×224
- Convert pixel values to [0, 1]
- ImageNet normalization
- No lesion removal
- No hair removal
- No additional color processing

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
| ISIC_9922133.jpg | 52.549 | 10.834 | 20.579 | 7.061 | 6 |
| ISIC_9922430.jpg | 54.245 | 9.733 | 19.046 | 12.566 | 5 |
| ISIC_9923018.jpg | 66.329 | 4.847 | 15.025 | 47.383 | 2 |
| ISIC_9886540.jpg | 55.163 | 11.486 | 15.172 | 18.794 | 5 |
| ISIC_9991451.jpg | 49.293 | 13.103 | 18.086 | -2.238 | 6 |
| ISIC_0068279.jpg | 45.629 | 10.655 | 16.081 | -15.207 | 6 |

## Observations
- TODO

## Decision / Next Experiment
- TODO
