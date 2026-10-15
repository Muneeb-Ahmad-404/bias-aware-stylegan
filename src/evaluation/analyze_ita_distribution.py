
import argparse
import logging
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from src.features.ita_inference import ItaInference


logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"
DEFAULT_METADATA = ROOT / "data" / "integrated" / "metadata.csv"
OUTPUT_DIR = ROOT / "reports" / "ita"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def resolve_path(value, root=ROOT):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze predicted ITA for integrated melanoma images."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config_path = resolve_path(args.config)
    metadata_path = resolve_path(args.metadata)
    output_dir = resolve_path(args.output_dir)

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Integrated metadata not found: {metadata_path}"
        )

    metadata = pd.read_csv(metadata_path)

    required_columns = {"image_id", "image_path"}
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        raise ValueError(
            f"Metadata is missing columns: {sorted(missing_columns)}"
        )

    # Resolve image paths using the dataset directories from the config.
    dataset_dirs = {
        key: resolve_path(value["image_dir"])
        for key, value in config["datasets"].items()
    }

    def locate_image(raw_path):
        path = Path(str(raw_path)).expanduser()

        if path.is_absolute() and path.is_file():
            return path

        candidates = [resolve_path(raw_path)]

        # Integrated metadata may contain paths relative to a source image_dir.
        for image_dir in dataset_dirs.values():
            candidates.append((image_dir / path).resolve())

        # Some metadata records contain only an image filename or image ID.
        for image_dir in dataset_dirs.values():
            candidates.extend(
                image_dir.rglob(path.name)
                if not path.is_absolute()
                else []
            )

        for candidate in candidates:
            if candidate.is_file():
                return candidate

        return None

    inference = ItaInference(str(config_path))
    inference.load_models()

    results = []
    failures = []

    for index, row in metadata.iterrows():
        image_id = str(row["image_id"])
        image_path = locate_image(row["image_path"])

        if image_path is None:
            failures.append({
                "image_id": image_id,
                "image_path": row["image_path"],
                "reason": "Image file not found",
            })
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            failures.append({
                "image_id": image_id,
                "image_path": str(image_path),
                "reason": "Unsupported image extension",
            })
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            failures.append({
                "image_id": image_id,
                "image_path": str(image_path),
                "reason": "Image could not be decoded",
            })
            continue

        try:
            # Raw image inference: no E2/E3 preprocessing.
            l_value, a_value, b_value = inference.get_prediction(image)
            ita = inference.calculate_ita(l_value, b_value)

            result = {
                "image_id": image_id,
                "image_path": str(image_path),
                "l": l_value,
                "a": a_value,
                "b": b_value,
                "ita": ita,
            }

            # Preserve the original metadata labels and fields.
            for column in metadata.columns:
                if column not in result:
                    result[column] = row[column]

            results.append(result)

        except Exception as exc:
            logger.exception("Inference failed for image %s", image_id)
            failures.append({
                "image_id": image_id,
                "image_path": str(image_path),
                "reason": str(exc),
            })

        if (index + 1) % 100 == 0 or index + 1 == len(metadata):
            logger.info(
                "Rows processed: %d/%d | predictions: %d | failures: %d",
                index + 1,
                len(metadata),
                len(results),
                len(failures),
            )

    if not results:
        raise RuntimeError("No integrated images were successfully analyzed.")

    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.DataFrame(results)
    predictions_path = output_dir / "integrated_ita_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    ita_values = predictions["ita"].astype(float)
    summary = {
        "images_in_metadata": len(metadata),
        "images_predicted": len(predictions),
        "images_failed": len(failures),
        "ita_mean": ita_values.mean(),
        "ita_std": ita_values.std(),
        "ita_min": ita_values.min(),
        "ita_q10": ita_values.quantile(0.10),
        "ita_q25": ita_values.quantile(0.25),
        "ita_median": ita_values.median(),
        "ita_q75": ita_values.quantile(0.75),
        "ita_q90": ita_values.quantile(0.90),
        "ita_max": ita_values.max(),
    }

    pd.DataFrame([summary]).to_csv(
        output_dir / "integrated_ita_summary.csv",
        index=False,
    )

    if failures:
        pd.DataFrame(failures).to_csv(
            output_dir / "integrated_ita_failures.csv",
            index=False,
        )

    logger.info("Predictions: %s", predictions_path)
    logger.info("Summary: %s", output_dir / "integrated_ita_summary.csv")


if __name__ == "__main__":
    main()
