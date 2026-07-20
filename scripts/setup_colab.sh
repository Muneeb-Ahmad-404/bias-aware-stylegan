#!/usr/bin/env bash

set -e

echo "Installing dependencies..."

python -m pip install --upgrade uv
uv sync

echo "Configuring Git..."

git config --global user.name "$GITHUB_USERNAME"
git config --global user.email "$GITHUB_EMAIL"

git remote set-url origin https://${GITHUB_TOKEN}@github.com/Muneeb-Ahmad-404/bias-aware-stylegan.git


echo "Configuring DVC remote..."

uv run dvc remote modify --local origin_processed auth basic
uv run dvc remote modify --local origin_processed user "$DAGSHUB_USERNAME"
uv run dvc remote modify --local origin_processed password "$DAGSHUB_TOKEN"

echo "Creating directories..."

mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/cleaned
mkdir -p data/splits


echo "Verifying setup..."

git remote -v
uv run dvc --version
uv run dvc remote list

echo "Setup completed successfully."