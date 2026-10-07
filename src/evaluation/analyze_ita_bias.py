from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


RESULTS_PATH = Path(
    "reports/ita/mskcc_validation_results.csv"
)

OUTPUT_PATH = Path(
    "reports/ita/mskcc_bias_by_ita_range.csv"
)


def analyze_experiment(df, experiment):
    """Analyze prediction error across reference ITA quartiles."""

    prediction_column = f"{experiment}_ita"

    data = df.dropna(
        subset=["reference_ita", prediction_column]
    ).copy()

    data["error"] = (
        data[prediction_column]
        - data["reference_ita"]
    )

    # Divide reference ITA into four equally sized groups
    data["ita_quartile"] = pd.qcut(
        data["reference_ita"],
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    )

    results = []

    for quartile, group in data.groupby(
        "ita_quartile",
        observed=True,
    ):
        y_true = group["reference_ita"].to_numpy()
        y_pred = group[prediction_column].to_numpy()

        results.append({
            "experiment": experiment.upper(),
            "ita_range": quartile,
            "n": len(group),
            "reference_min": y_true.min(),
            "reference_max": y_true.max(),
            "reference_mean": y_true.mean(),
            "bias": np.mean(y_pred - y_true),
            "mae": mean_absolute_error(
                y_true,
                y_pred,
            ),
            "rmse": np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            ),
        })

    return results


def main():
    # ---------------------------------------------------------
    # Load validation results
    # ---------------------------------------------------------

    df = pd.read_csv(RESULTS_PATH)

    # ---------------------------------------------------------
    # Analyze each experiment
    # ---------------------------------------------------------

    results = []

    for experiment in ["e1", "e2", "e3"]:
        results.extend(
            analyze_experiment(
                df,
                experiment,
            )
        )

    results_df = pd.DataFrame(results)

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ---------------------------------------------------------
    # Display results
    # ---------------------------------------------------------

    print("\nMSKCC ITA Bias by Reference ITA Range")
    print("======================================")

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    print(
        f"\nResults saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()