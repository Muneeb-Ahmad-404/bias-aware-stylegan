# Architecture & Design Decision Log

## [2026-06-22] - Milestone 1: Engineering Foundation & Reproducibility Scaffold

### Repository Skeleton [Done]

* **Decision:** Implemented a modular repository structure separating concerns across clear directory boundaries.

```text
main repo
   ├── data
   │   ├── input
   │   │   ├── fitzpatrick
   │   │   └── isic
   │   └── output
   └── src
       ├── config
       ├── classifier
       ├── style2gan
       ├── evaluation
       └── reports

```

* **Justification:** Ensures configuration, data assets, source logic, and validation modules remain distinct and transferable. The `src/reports/` directory isolates scripts that generate phase-specific performance metrics under changing parameters, enabling evidence-based configuration tracking.

### Docker [Todo]

* **Decision:** Containerization strategy selected for local and remote deployment.
* **Justification:** Guarantees a standardized runtime environment across development platforms.

### CI (Continuous Integration) [Todo]

* **Decision:** Implementation of automated tests prior to main trunk merging.
* **Justification:** Automatically catches integration and operational failures before runtime code is deployed.

### DVC (Data Version Control) [Todo-Later]

* **Decision:** Data tracking isolated from Git history via pointer tracking.
* **Justification:** Provides complete version control for massive dataset states without bloating the remote repository.

### Experiment Tracking [Todo]

* **Decision:** Hybrid reporting utilizing local output scripts paired with Weights & Biases (`wandb`).
* **Justification:** Uses local code to finalize concrete summary profiles, while leveraging `wandb` real-time streams to monitor metrics and manually halt inefficient or collapsing training runs early.

---

## [2026-06-22] - Milestone 2: Data Ingestion & Profiling

### 1. Ingestion of Base Metadata Schema

* **Decision:** Locally staged `fitzpatrick17k.csv` and `ISIC_2020_Training_GroundTruth.csv` within input subdirectories.
* **Justification:** Establishes data schema locally without loading images, enables rapid development and testing of mapping functions on test rows.