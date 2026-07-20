#!/bin/bash

set -e

echo "Installing dependencies..."

pip install uv
uv sync


echo "Configuring Git..."

git config --global user.name "$GITHUB_USERNAME"
git config --global user.email "$GITHUB_EMAIL"

git remote set-url origin https://${GITHUB_TOKEN}@github.com/Muneeb-Ahmad-404/bias-aware-stylegan.git


echo "Configuring DVC remote..."

dvc remote modify --local origin_raw auth basic
dvc remote modify --local origin_raw user "$DAGSHUB_USERNAME"
dvc remote modify --local origin_raw password "$DAGSHUB_TOKEN"


echo "Creating project directories..."

mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/cleaned
mkdir -p data/splits
mkdir -p reports


echo "Verifying setup..."

echo "Git remote:"
git remote -v

echo "DVC remotes:"
dvc remote list


echo "Colab setup completed successfully."