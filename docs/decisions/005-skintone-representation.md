# Decision 005: Skin-Tone Representation

**Status:** Under Investigation
**Date:** 2026-09-16

## Context

The project aims to generate melanoma images across the full skin-tone spectrum. A representation of skin tone is therefore required for dataset analysis and potentially for GAN conditioning.

## Options Considered

* **Fitzpatrick I–VI:** Clinically established and already available in Fitzpatrick17k, but categorical and not directly equivalent to image pigmentation.
* **ITA:** Continuous, image-derived representation that may provide finer control over skin-tone variation, but its reliability depends on image characteristics and acquisition conditions.
* **Learned representation:** Potentially adaptable to the domain, but adds substantial complexity and requires additional training and validation.

## Decision

Investigate **ITA as the primary experimental skin-tone representation**.

A pretrained dermatoscopic colorimetry model will be used to estimate ITA. The five available fold models will initially be evaluated as an ensemble, following the original model's inference methodology where applicable.

ITA will be stored as a **continuous score**, not converted directly into Fitzpatrick I–VI labels.

## Rationale

ITA provides a continuous representation that is potentially better suited to controlling gradual skin-tone variation during image generation. However, its suitability for the project's ISIC melanoma images must be experimentally validated.

## Validation

Before adopting ITA for GAN conditioning:

* Verify the pretrained model's required preprocessing.
* Test predictions on a sample of ISIC melanoma images.
* Examine consistency across the five models.
* Analyze the resulting ITA distribution.
* Compare against available Fitzpatrick-labeled data where possible.
* Visually inspect representative samples across the ITA range.

## Alternatives Deferred

Fitzpatrick-based conditioning and other representations are retained as alternatives and may be evaluated later if ITA proves unsuitable or if comparison is scientifically useful.
