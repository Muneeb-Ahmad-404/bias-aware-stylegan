"""Fairness metrics per ITA skin-tone bin (FR9 / FR10).

Computes sensitivity, specificity, accuracy, and ROC-AUC overall and per ITA
group, plus the lightest-to-darkest performance gap for comparing a baseline
classifier against a synthetic-augmented classifier.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from src.evaluation.ita import ITA_BINS


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    n = int(len(y_true))
    if n == 0:
        return {"count": 0, "positives": 0, "sensitivity": None, "specificity": None, "accuracy": None, "roc_auc": None}

    y_pred = (y_prob >= threshold).astype(int)
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    sensitivity = round(tp / positives, 4) if positives > 0 else None
    specificity = round(tn / negatives, 4) if negatives > 0 else None
    accuracy = round(float(accuracy_score(y_true, y_pred)), 4)
    roc_auc = (
        round(float(roc_auc_score(y_true, y_prob)), 4)
        if positives > 0 and negatives > 0
        else None
    )

    return {
        "count": n,
        "positives": positives,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "roc_auc": roc_auc,
    }


def evaluate_by_ita_group(
    df_preds: pd.DataFrame,
    y_true_col: str = "y_true",
    y_prob_col: str = "y_prob",
    group_col: str = "skin_tone_group",
    threshold: float = 0.5,
) -> dict:
    """Evaluate classifier predictions per ITA bin and compute light-vs-dark gaps."""
    y_all = df_preds[y_true_col].to_numpy(dtype=int)
    p_all = df_preds[y_prob_col].to_numpy(dtype=float)

    per_group = {}
    for group_name, _, _ in ITA_BINS:
        sub = df_preds[df_preds[group_col] == group_name]
        per_group[group_name] = _binary_metrics(
            sub[y_true_col].to_numpy(dtype=int),
            sub[y_prob_col].to_numpy(dtype=float),
            threshold=threshold,
        )

    light_mask = df_preds[group_col].isin(["Very Light (I)", "Light (II)"])
    dark_mask = df_preds[group_col].isin(["Brown (V)", "Dark (VI)"])
    tan_dark_mask = df_preds[group_col].isin(["Tan (IV)", "Brown (V)", "Dark (VI)"])

    light_m = _binary_metrics(
        df_preds.loc[light_mask, y_true_col].to_numpy(dtype=int),
        df_preds.loc[light_mask, y_prob_col].to_numpy(dtype=float),
        threshold=threshold,
    )
    dark_m = _binary_metrics(
        df_preds.loc[dark_mask, y_true_col].to_numpy(dtype=int),
        df_preds.loc[dark_mask, y_prob_col].to_numpy(dtype=float),
        threshold=threshold,
    )
    tan_dark_m = _binary_metrics(
        df_preds.loc[tan_dark_mask, y_true_col].to_numpy(dtype=int),
        df_preds.loc[tan_dark_mask, y_prob_col].to_numpy(dtype=float),
        threshold=threshold,
    )

    def _gap(a: float | None, b: float | None) -> float | None:
        return round(a - b, 4) if (a is not None and b is not None) else None

    return {
        "overall": _binary_metrics(y_all, p_all, threshold=threshold),
        "per_ita_group": per_group,
        "aggregates": {
            "light_I_II": light_m,
            "dark_V_VI": dark_m,
            "tan_to_dark_IV_VI": tan_dark_m,
        },
        "gaps": {
            "light_minus_dark_sensitivity": _gap(light_m["sensitivity"], dark_m["sensitivity"]),
            "light_minus_dark_roc_auc": _gap(light_m["roc_auc"], dark_m["roc_auc"]),
            "light_minus_tan_dark_sensitivity": _gap(light_m["sensitivity"], tan_dark_m["sensitivity"]),
            "light_minus_tan_dark_roc_auc": _gap(light_m["roc_auc"], tan_dark_m["roc_auc"]),
        },
    }


def save_fairness_report(report: dict, output_path: str | Path) -> Path:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out
