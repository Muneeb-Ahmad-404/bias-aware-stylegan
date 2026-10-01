import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features.ita_inference import ItaInference


# =============================================================
# Logging
# =============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# =============================================================
# Paths
# =============================================================

IMAGE_DIR = Path("data/validator/mskcc/images")
CSV_PATH = Path("data/validator/mskcc/s7.csv")
MODEL_CONFIG = "configs/config.yaml"

OUTPUT_PATH = Path("reports/ita/mskcc_validation_results.csv")
SUMMARY_PATH = Path("reports/ita/mskcc_validation_summary.csv")


# =============================================================
# Execution settings
# =============================================================

# CPU environments:
# Increase this if your machine has enough CPU cores.
CPU_WORKERS = max(1, min(8, os.cpu_count() or 1))

# GPU:
# Keep one worker because ItaInference contains the model(s).
GPU_WORKERS = 1


# =============================================================
# Metrics
# =============================================================

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "n": len(y_true),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "pearson_r": pearsonr(y_true, y_pred).statistic,
        "spearman_rho": spearmanr(y_true, y_pred).statistic,
        "bias": np.mean(y_pred - y_true),
        "std_error": np.std(y_pred - y_true, ddof=1),
    }


# =============================================================
# Single-image processing
# =============================================================

def process_image(img_path, reference_df, ita):
    """
    Process one image through E1, E2 and E3.

    Returns:
        dict containing results, or None if the image should be skipped.
    """

    isic_id = img_path.stem

    # ---------------------------------------------------------
    # Check reference data
    # ---------------------------------------------------------

    if isic_id not in reference_df.index:
        logger.warning(
            "%s not found in MSKCC reference CSV",
            isic_id,
        )
        return None

    img_bgr = None
    hair_free_img = None
    clean_patch = None

    try:
        # -----------------------------------------------------
        # Load image
        # -----------------------------------------------------

        img_bgr = cv.imread(str(img_path))

        if img_bgr is None:
            logger.error(
                "Could not read image: %s",
                img_path,
            )
            return None

        row = reference_df.loc[isic_id]

        reference_ita = float(row["average_ita"])

        result = {
            "isic_id": isic_id,
            "reference_ita": reference_ita,
        }

        # -----------------------------------------------------
        # Metadata
        # -----------------------------------------------------

        for column in [
            "dermoscopic_type",
            "anatomic_site",
            "tag_id",
            "img_ita",
        ]:
            if column in row.index:
                result[column] = row[column]

        # =====================================================
        # E1: Raw image
        # =====================================================

        l, a, b = ita.get_prediction(img_bgr)
        e1_ita = ita.calculate_ita(l, b)

        result["e1_l"] = l
        result["e1_a"] = a
        result["e1_b"] = b
        result["e1_ita"] = e1_ita

        # =====================================================
        # E2: Multi-scale hair removal
        # =====================================================

        hair_free_img = ita.remove_hair_multiscale(img_bgr)

        l, a, b = ita.get_prediction(hair_free_img)
        e2_ita = ita.calculate_ita(l, b)

        result["e2_l"] = l
        result["e2_a"] = a
        result["e2_b"] = b
        result["e2_ita"] = e2_ita

        # =====================================================
        # E3: Hair removal + clean patch
        # =====================================================

        clean_patch = ita.extract_clean_patch(hair_free_img)

        l, a, b = ita.get_prediction(clean_patch)
        e3_ita = ita.calculate_ita(l, b)

        result["e3_l"] = l
        result["e3_a"] = a
        result["e3_b"] = b
        result["e3_ita"] = e3_ita

        return result

    except Exception:
        logger.exception(
            "Failed processing %s",
            isic_id,
        )
        return None

    finally:
        # Release references before the next image
        img_bgr = None
        hair_free_img = None
        clean_patch = None


# =============================================================
# Validation
# =============================================================

def validate_mskcc():

    logger.info("Starting MSKCC ITA validation")

    # ---------------------------------------------------------
    # Detect environment
    # ---------------------------------------------------------

    use_gpu = torch.cuda.is_available()

    if use_gpu:
        device_name = torch.cuda.get_device_name(0)
        workers = GPU_WORKERS

        logger.info(
            "CUDA available: %s",
            device_name,
        )
        logger.info(
            "Using %d worker for GPU inference",
            workers,
        )

    else:
        workers = CPU_WORKERS

        logger.info("CUDA unavailable")
        logger.info(
            "Using CPU with %d workers",
            workers,
        )

    # ---------------------------------------------------------
    # Load reference CSV
    # ---------------------------------------------------------

    logger.info(
        "Loading MSKCC reference CSV: %s",
        CSV_PATH,
    )

    df = pd.read_csv(CSV_PATH)

    required_columns = {
        "isic_id",
        "average_ita",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns in {CSV_PATH}: {missing}"
        )

    df = df.dropna(
        subset=["average_ita"]
    ).copy()

    df["isic_id"] = df["isic_id"].astype(str)

    df = df.set_index("isic_id")

    logger.info(
        "Loaded %d reference records",
        len(df),
    )

    # ---------------------------------------------------------
    # Find images
    # ---------------------------------------------------------

    image_paths = sorted(
        path
        for path in IMAGE_DIR.iterdir()
        if path.suffix.lower()
        in {".jpg", ".jpeg", ".png"}
    )

    logger.info(
        "Found %d images in %s",
        len(image_paths),
        IMAGE_DIR,
    )

    # ---------------------------------------------------------
    # Load model once
    # ---------------------------------------------------------

    logger.info("Loading ITA models")

    ita = ItaInference(MODEL_CONFIG)
    ita.load_models()

    logger.info("ITA models loaded")

    # ---------------------------------------------------------
    # Process images
    # ---------------------------------------------------------

    results = []

    total = len(image_paths)

    if use_gpu:
        # -----------------------------------------------------
        # GPU MODE
        #
        # Keep a single ItaInference instance.
        # Do not create multiple GPU model copies.
        # -----------------------------------------------------

        for index, img_path in enumerate(
            image_paths,
            start=1,
        ):

            result = process_image(
                img_path,
                df,
                ita,
            )

            if result is not None:
                results.append(result)

                logger.info(
                    "[%d/%d] %s | ref=%.2f | "
                    "E1=%.2f | E2=%.2f | E3=%.2f",
                    index,
                    total,
                    result["isic_id"],
                    result["reference_ita"],
                    result["e1_ita"],
                    result["e2_ita"],
                    result["e3_ita"],
                )

    else:
        # -----------------------------------------------------
        # CPU MODE
        #
        # Process multiple images concurrently.
        # -----------------------------------------------------

        logger.info(
            "Starting parallel CPU processing with %d workers",
            workers,
        )

        with ThreadPoolExecutor(
            max_workers=workers
        ) as executor:

            futures = {
                executor.submit(
                    process_image,
                    img_path,
                    df,
                    ita,
                ): img_path
                for img_path in image_paths
            }

            completed = 0

            for future in as_completed(futures):

                completed += 1

                img_path = futures[future]

                try:
                    result = future.result()

                    if result is not None:
                        results.append(result)

                        logger.info(
                            "[%d/%d] %s | ref=%.2f | "
                            "E1=%.2f | E2=%.2f | E3=%.2f",
                            completed,
                            total,
                            result["isic_id"],
                            result["reference_ita"],
                            result["e1_ita"],
                            result["e2_ita"],
                            result["e3_ita"],
                        )

                except Exception:
                    logger.exception(
                        "Worker failed for %s",
                        img_path,
                    )

    # ---------------------------------------------------------
    # Check results
    # ---------------------------------------------------------

    logger.info(
        "Processing complete: %d/%d images produced results",
        len(results),
        total,
    )

    if not results:
        raise RuntimeError(
            "No validation results were generated."
        )

    # ---------------------------------------------------------
    # Save detailed results
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    logger.info(
        "Detailed results saved to: %s",
        OUTPUT_PATH,
    )

    # ---------------------------------------------------------
    # Calculate summary
    # ---------------------------------------------------------

    summary_rows = []

    for experiment in [
        "e1",
        "e2",
        "e3",
    ]:

        valid = results_df[
            results_df["reference_ita"].notna()
            & results_df[
                f"{experiment}_ita"
            ].notna()
        ]

        if len(valid) < 2:
            logger.warning(
                "Not enough valid samples for %s",
                experiment.upper(),
            )
            continue

        metrics = calculate_metrics(
            valid["reference_ita"],
            valid[f"{experiment}_ita"],
        )

        metrics["experiment"] = experiment.upper()

        summary_rows.append(metrics)

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df = summary_df[
        [
            "experiment",
            "n",
            "mae",
            "rmse",
            "pearson_r",
            "spearman_rho",
            "bias",
            "std_error",
        ]
    ]

    # ---------------------------------------------------------
    # Save summary
    # ---------------------------------------------------------

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    logger.info(
        "Summary saved to: %s",
        SUMMARY_PATH,
    )

    # ---------------------------------------------------------
    # Final output
    # ---------------------------------------------------------

    print("\nValidation complete.")
    print(
        f"Detailed results: {OUTPUT_PATH}"
    )
    print(
        f"Summary results:  {SUMMARY_PATH}"
    )
    print("\nSummary:")
    print(
        summary_df.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    validate_mskcc()