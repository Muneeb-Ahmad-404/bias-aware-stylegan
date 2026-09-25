#!/usr/bin/env bash

set -e

URL="https://isic-archive.s3.amazonaws.com/dois/10.34970-962049/mskcc-skin-tone-labeling-dataset.zip"
VALIDATOR_DIR="data/validator"
ZIP_PATH="${VALIDATOR_DIR}/mskcc-skin-tone-labeling-dataset.zip"
EXTRACT_DIR="${VALIDATOR_DIR}/mskcc"

echo "Creating validator directory..."
mkdir -p "$VALIDATOR_DIR"

echo "Downloading MSKCC Skin Tone Labeling Dataset..."
curl -L --fail --retry 3 -o "$ZIP_PATH" "$URL"

echo "Extracting dataset..."
rm -rf "$EXTRACT_DIR"
mkdir -p "$EXTRACT_DIR"

unzip -q "$ZIP_PATH" -d "$EXTRACT_DIR"

echo "Locating S7..."
S7_FILE=$(find "$EXTRACT_DIR" -type f -iname 's7.csv' | head -n 1)

if [[ ! -f "$S7_FILE" ]]; then
    echo "ERROR: $S7_FILE not found."
    exit 1
fi

cp "$S7_FILE" "$EXTRACT_DIR/"

echo "Cleaning up archive..."
rm "$ZIP_PATH"

echo "MSKCC dataset setup completed."
echo "Dataset: $EXTRACT_DIR"
echo "S7: $EXTRACT_DIR/$(basename "$S7_FILE")"