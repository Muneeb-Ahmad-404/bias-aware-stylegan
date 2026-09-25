import logging as logger
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features.ita_inference import ItaInference


IMAGE_DIR = Path("data/validator/mskcc/images")
CSV_PATH = Path("data/validator/mskcc/s7.csv")
MODEL_CONFIG = "configs/config.yaml"

OUTPUT_PATH = Path(
    "reports/ita/mskcc_validation_results.csv"
)

SUMMARY_PATH = Path(
    "reports/ita/mskcc_validation_summary.csv"
)


def calculate_metrics(y_true, y_pred):
    """Calculate regression/agreement metrics."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "n": len(y_true),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(
            mean_squared_error(y_true, y_pred)
        ),
        "pearson_r": pearsonr(
            y_true, y_pred
        ).statistic,
        "spearman_rho": spearmanr(
            y_true, y_pred
        ).statistic,
        "bias": np.mean(y_pred - y_true),
        "std_error": np.std(
            y_pred - y_true,
            ddof=1,
        ),
    }


def validate_mskcc():
    # ---------------------------------------------------------
    # Load MSKCC reference data
    # ---------------------------------------------------------

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

    # Only images with a colorimeter ITA reference
    df = df.dropna(
        subset=["average_ita"]
    ).copy()

    df["isic_id"] = df["isic_id"].astype(str)
    df = df.set_index("isic_id")

    logger.info(
        "MSKCC reference rows with ITA: %d",
        len(df),
    )

    # ---------------------------------------------------------
    # Load existing pretrained ITA model
    # ---------------------------------------------------------

    ita = ItaInference(MODEL_CONFIG)
    ita.load_models()
  
    # ---------------------------------------------------------
    # Run E1 / E2 / E3
    # ---------------------------------------------------------

    results = []

    for img_name, img_bgr in ita.load_images(IMAGE_DIR):

        isic_id = Path(img_name).stem

        if isic_id not in df.index:
            logger.warning(
                "%s: not found in MSKCC reference CSV",
                isic_id,
            )
            continue

        row = df.loc[isic_id]

        try:
            reference_ita = float(row["average_ita"])

            result = {
                "isic_id": isic_id,
                "reference_ita": reference_ita,
            }

            # Preserve useful metadata if available
            for column in ["type", "anatomic_site"]:
                if column in row.index:
                    result[column] = row[column]

            # E1: Raw image
            try:
                l, a, b = ita.get_prediction(img_bgr)
                predicted_ita = ita.calculate_ita(l, b)

                result.update({
                    "e1_l": l,
                    "e1_a": a,
                    "e1_b": b,
                    "e1_ita": predicted_ita,
                })

            except Exception:
                logger.exception("%s: E1 failed", isic_id)

            # E2: Multiscale hair removal
            hair_free_img = None

            try:
                hair_free_img = ita.remove_hair_multiscale(img_bgr)

                l, a, b = ita.get_prediction(hair_free_img)
                predicted_ita = ita.calculate_ita(l, b)

                result.update({
                    "e2_l": l,
                    "e2_a": a,
                    "e2_b": b,
                    "e2_ita": predicted_ita,
                })

            except Exception:
                logger.exception("%s: E2 failed", isic_id)

            # E3: Hair removal + clean patch
            try:
                if hair_free_img is None:
                    raise RuntimeError(
                        "Hair removal failed; cannot run E3"
                    )

                clean_patch = ita.extract_clean_patch(
                    hair_free_img
                )

                if clean_patch is None:
                    raise RuntimeError(
                        "Could not extract clean patch"
                    )

                l, a, b = ita.get_prediction(clean_patch)
                predicted_ita = ita.calculate_ita(l, b)

                result.update({
                    "e3_l": l,
                    "e3_a": a,
                    "e3_b": b,
                    "e3_ita": predicted_ita,
                })

            except Exception:
                logger.exception("%s: E3 failed", isic_id)

            results.append(result)

            logger.info(
                "%s | reference=%.2f | E1=%.2f | "
                "E2=%.2f | E3=%.2f",
                isic_id,
                reference_ita,
                result.get("e1_ita", np.nan),
                result.get("e2_ita", np.nan),
                result.get("e3_ita", np.nan),
            )

        except Exception:
            logger.exception(
                "%s: validation failed",
                isic_id,
            )

    # ---------------------------------------------------------
    # Save per-image results
    # ---------------------------------------------------------

    results_df = pd.DataFrame(results)

    if results_df.empty:
        raise RuntimeError(
            "No images could be validated."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Calculate summary metrics
    # ---------------------------------------------------------

    summary = []

    for experiment in [
        "e1",
        "e2",
        "e3",
    ]:

        prediction_column = f"{experiment}_ita"

        valid = results_df.dropna(
            subset=[
                "reference_ita",
                prediction_column,
            ]
        )

        if valid.empty:
            logger.warning(
                "%s: no valid predictions",
                experiment.upper(),
            )
            continue

        y_true = valid[
            "reference_ita"
        ].to_numpy()

        y_pred = valid[
            prediction_column
        ].to_numpy()

        metrics = calculate_metrics(
            y_true,
            y_pred,
        )

        metrics["experiment"] = experiment.upper()

        summary.append(metrics)

    summary_df = pd.DataFrame(summary)

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

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------

    print("\nMSKCC ITA Validation")
    print("====================")
    print(
        f"Images evaluated   : {len(results_df)}"
    )

    print("\nResults:")
    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print(
        f"\nDetailed results: {OUTPUT_PATH}"
    )

    print(
        f"Summary results : {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    validate_mskcc()