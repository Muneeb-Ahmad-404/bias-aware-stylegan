# Experiment 002 — Multiscale Hair Removal

## Objective

Evaluate the effect of multiscale hair removal on skin-tone estimation using the pretrained five-fold Lab regression models.

## Date

2026-09-20 19:47:39

## Input

- Dataset: ISIC 2020
- Subset: current test images
- Images processed: 6
- Image directory: `data/raw/isic2020/train`
- Original resolution: 512×512

## Change from Previous Experiment

Multiscale hair removal was introduced before model inference. All other inference steps remain unchanged.

## Preprocessing

- BGR → RGB
- Multiscale blackhat hair detection
- Hair mask dilation
- Telea inpainting
- Resize to 224×224
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
| ISIC_9922133.jpg | 52.578 | 9.671 | 19.229 | 7.637 | 6 |
| ISIC_9922430.jpg | 57.718 | 8.696 | 18.471 | 22.677 | 4 |
| ISIC_9923018.jpg | 66.915 | 4.945 | 15.096 | 48.252 | 2 |
| ISIC_9886540.jpg | 53.991 | 10.603 | 14.879 | 15.014 | 5 |
| ISIC_9991451.jpg | 50.706 | 12.196 | 18.183 | 2.224 | 6 |
| ISIC_0068279.jpg | 52.712 | 12.256 | 15.601 | 9.860 | 6 |

## Observations

TODO

## Decision / Next Experiment

TODO
