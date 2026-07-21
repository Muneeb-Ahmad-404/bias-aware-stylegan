# Google Colab Setup

This guide describes how to configure the project for running the pipeline on Google Colab.

## Prerequisites

* GitHub account with access to this repository.
* DagsHub account with access to the configured DVC remote.
* Copy of `.env.example`.

---

## 1. Configure Colab Secrets

Open the **Secrets** panel in Google Colab (key icon in the left sidebar) and add the variables listed in `.env.example`:

* `GITHUB_USERNAME`
* `GITHUB_EMAIL`
* `GITHUB_TOKEN`
* `DAGSHUB_USERNAME`
* `DAGSHUB_TOKEN`

Enable notebook access for each secret after adding it.

---

## 2. Clone the Repository

```bash
git clone https://github.com/Muneeb-Ahmad-404/bias-aware-stylegan.git
cd bias-aware-stylegan
```

---

## 3. Run the Setup Script

```bash
python scripts/setup.py
```

The setup script automatically:

* Installs project dependencies using `uv`.
* Configures Git authentication.
* Configures the DVC remote.
* Creates the required project directories.
* Verifies the Git and DVC configuration.

---

## 4. Verify the Setup

```bash
git remote -v
```

```bash
dvc remote list
```

If both commands execute successfully, the Colab environment is ready for running the project pipeline.
