# Experiment 004_patch_color_analysis — Patch Color Analysis

## Objective
Analyze the pixel-level color characteristics of extracted surrounding-skin patches.

## Date
2026-09-20 22:49:09

## Input
- Dataset: ISIC 2020
- Images processed: 1
- Image directory: `data/raw/isic2020/train`

## Preprocessing
- Hair removal using multiscale black-hat detection and inpainting
- Surrounding-skin patch extraction
- RGB mean and median calculation
- OpenCV LAB mean and median calculation

## Results

| image | rgb_mean_r | rgb_mean_g | rgb_mean_b | rgb_median_r | rgb_median_g | rgb_median_b | opencv_lab_mean_l | opencv_lab_mean_a | opencv_lab_mean_b | opencv_lab_median_l | opencv_lab_median_a | opencv_lab_median_b |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ISIC_9922133.jpg | 159.162 | 126.406 | 104.309 | 159.000 | 127.000 | 105.000 | 141.646 | 137.396 | 144.729 | 142.000 | 137.000 | 145.000 |

## Observations
- TODO

## Next Experiment
- TODO
