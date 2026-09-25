import logging as logger
from pathlib import Path

import cv2 as cv
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.features.ita_inference import ItaInference


IMAGE_DIR = Path("data/validator/mskcc/images")
CSV_PATH = Path("data/validator/mskcc/s7.csv")
MODEL_CONFIG = "configs/config.yaml"

OUTPUT_PATH = Path(
    "reports/ita/mskcc_validation_results.csv"
)


def validate_mskcc():
    # Load MSKCC colorimeter/ITA data
    df = pd.read_csv(CSV_PATH)

    # Only rows with a colorimeter reference
    df = df.dropna(subset=["average_ita"])

    # Index CSV by ISIC ID for fast lookup
    df = df.set_index("isic_id")

    # Use your existing inference implementation
    ita = ItaInference(MODEL_CONFIG)
    ita.load_models()

    results = []

    images = sorted(IMAGE_DIR.glob("*"))

    logger.info("Found %d downloaded images", len(images))

    for image_path in images:

        # Example:
        # ISIC_1234567.jpg -> ISIC_1234567
        isic_id = image_path.stem

        # Check whether this downloaded image exists in
        # the MSKCC ITA CSV
        if isic_id not in df.index:
            logger.warning(
                "%s: not found in MSKCC ITA CSV",
                isic_id,
            )
            continue

        row = df.loc[isic_id]

        try:
            img_bgr = cv.imread(str(image_path))

            if img_bgr is None:
                logger.warning(
                    "%s: could not read image",
                    isic_id,
                )
                continue

            # ---- Existing inference pipeline ----

            hair_free_img = ita.remove_hair_multiscale(img_bgr)

            clean_patch = ita.extract_clean_patch(hair_free_img)

            if clean_patch is None:
                logger.warning(
                    "%s: could not extract clean patch",
                    isic_id,
                )
                continue

            l, a, b = ita.get_prediction(clean_patch)

            predicted_ita = ita.calculate_ita(l, b)

            # -------------------------------------

            reference_ita = float(row["average_ita"])

            results.append({
                "isic_id": isic_id,
                "type": row["type"],
                "anatomic_site": row["anatomic_site"],
                "predicted_l": l,
                "predicted_a": a,
                "predicted_b": b,
                "predicted_ita": predicted_ita,
                "reference_ita": reference_ita,
                "error": predicted_ita - reference_ita,
                "absolute_error": abs(
                    predicted_ita - reference_ita
                ),
            })

            logger.info(
                "%s: predicted ITA=%.2f, "
                "reference ITA=%.2f",
                isic_id,
                predicted_ita,
                reference_ita,
            )

        except Exception as e:
            logger.exception(
                "%s: validation failed: %s",
                isic_id,
                e,
            )

    results_df = pd.DataFrame(results)

    if results_df.empty:
        raise RuntimeError(
            "No downloaded images could be validated."
        )

    y_true = results_df["reference_ita"].to_numpy()
    y_pred = results_df["predicted_ita"].to_numpy()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    pearson = pearsonr(
        y_true,
        y_pred,
    ).statistic

    spearman = spearmanr(
        y_true,
        y_pred,
    ).statistic

    print("\nMSKCC ITA Validation")
    print("====================")
    print(f"Images evaluated : {len(results_df)}")
    print(f"MAE              : {mae:.3f}")
    print(f"RMSE             : {rmse:.3f}")
    print(f"Pearson r        : {pearson:.3f}")
    print(f"Spearman rho     : {spearman:.3f}")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        f"\nDetailed results saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    validate_mskcc()