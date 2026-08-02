# ADR-004: Melanoma Dataset Integration Strategy

## Status
Accepted

## Context

The project focuses on melanoma-specific image synthesis.
ISIC 2020 provides melanoma samples but lacks Fitzpatrick labels.
Fitzpatrick17k provides skin tone annotations and contains melanoma cases.

## Decision

Only melanoma samples are integrated from both datasets.

The unified metadata schema contains:

- image_id
- image_path
- fitzpatrick_label

Disease labels and malignancy labels are omitted because all samples represent melanoma.

## Consequences

The integrated dataset contains melanoma images only.
ISIC samples will receive Fitzpatrick labels during the ITA stage.
Fitzpatrick17k labels are retained for existing annotations.