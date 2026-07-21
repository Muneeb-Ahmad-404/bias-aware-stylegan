# ADR-003: Dataset Selection Strategy

## Status
Accepted

## Context

The problem we are trying to solve is the lack of available melanoma datasets with adequate representation of darker skin tones. For that purpose, we need a dataset that represents melanoma and a dataset that provides Fitzpatrick skin type information. However, no single dataset in our knowledge provides both.

For that purpose, we decided to use ISIC 2020, which is a large publicly available melanoma dataset, and Fitzpatrick17k, which is one of the few publicly available datasets that provides Fitzpatrick skin type labels and represents darker skin tones better than many commonly used dermatology datasets.

We plan to use dual conditioning: one condition for the melanoma lesion and the other for the skin type. Using these two datasets together better supports the goal of the project.

ISIC 2019 was also considered. However, since the project is specifically focused on melanoma, ISIC 2020 aligns better with the current scope. If the scope expands beyond melanoma, ISIC 2019 may become a more suitable choice because of its broader disease coverage.

## Benefits

- Publicly available datasets, supporting reproducibility.
- Strong melanoma representation through ISIC 2020.
- Fitzpatrick skin type labels through Fitzpatrick17k.
- Better support for fairness-related analysis and generation through dual conditioning.

## Limitations

- The metadata formats of both datasets are different.
- The modalities/domains of the datasets are different.
- ISIC 2020 does not provide explicit Fitzpatrick skin type annotations.
- Additional preprocessing is required before both datasets can be used together.

## Future Considerations

If the project scope expands to include multiple skin diseases, ISIC 2019 or other suitable datasets may be considered for broader disease coverage.

Similarly, if a better publicly available dataset becomes available with stronger skin tone representation or improved annotations, it may replace one or both of the selected datasets.