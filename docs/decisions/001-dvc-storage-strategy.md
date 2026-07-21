# ADR-001: Dataset Storage and Versioning Strategy

## Status
Accepted

## Context

The data is going to be passed through various intermediary steps in the pipeline. Later, different experiments may require changing these datasets and comparing results.

For this purpose, dataset versioning is necessary to keep track of different dataset states and ensure reproducibility throughout the project.

## Decision

DVC will be used for dataset version control.

The remote storage solution selected initially is DagsHub.

The storage workflow will be:

- GitHub:
  - Code
  - Configuration files
  - Documentation
  - DVC metadata files

- DVC + DagsHub:
  - Different dataset stages
  - Processed datasets
  - Experiment datasets
  - Generated artifacts

DagsHub was selected because separate repositories can be created for different stages, allowing easier access to specific datasets during development.

## Alternatives Considered

### AWS / Azure

AWS and Azure were considered as possible storage solutions.

They were skipped for now because of account registration and setup complexities.

They can be considered later to reduce manual management and provide a more scalable storage solution.

### Kaggle

Kaggle was considered because the initial datasets are obtained from Kaggle.

However, Kaggle does not provide proper dataset version control for the different pipeline stages, making it unsuitable as the main storage solution.

### DagsHub

DagsHub was selected because it integrates with DVC and provides an easier setup for initial development.

## Benefits

- Easy dataset versioning
- Easier experiment tracking
- Ability to compare different dataset versions
- Separation of code and large datasets

## Disadvantages

- Manual creation of DagsHub repositories
- Manual authorization steps
- Additional setup for collaborators

## Future Considerations

The project may switch to cloud storage solutions such as AWS or Azure later to reduce manual DagsHub management and improve scalability.