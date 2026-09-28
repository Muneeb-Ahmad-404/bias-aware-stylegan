#!/usr/bin/env bash

set -e

echo "Fetching ITA model..."

uv run dvc pull finetune_mskcc_blur0_lab

echo "ITA model fetched successfully."