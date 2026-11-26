"""Individual Typology Angle (ITA) skin-tone estimation for dermatology images.

Computes colorimetric ITA in CIE L*a*b* space on non-lesion background skin:
    ITA = arctan((L* - 50) / b*) * (180 / pi)

Reference:
    Chardon et al. (1991), Kinyanjui et al. (2020) "Fairness of Classifiers
    Across Skin Shades in Dermatology".
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# Standard Chardon / Kinyanjui ITA -> Fitzpatrick-proxy bins (in degrees)
ITA_BINS = [
    ("Very Light (I)", 55.0, float("inf")),
    ("Light (II)", 41.0, 55.0),
    ("Intermediate (III)", 28.0, 41.0),
    ("Tan (IV)", 10.0, 28.0),
    ("Brown (V)", -30.0, 10.0),
    ("Dark (VI)", float("-inf"), -30.0),
]


@dataclass
class ITAResult:
    """ITA measurement for a single image."""

    filename: str
    ita_deg: float
    skin_tone_group: str
    l_star_median: float
    b_star_median: float
    skin_pixel_ratio: float


def classify_ita(ita_deg: float) -> str:
    """Map an ITA angle (degrees) to its standard skin-tone category."""
    if math.isnan(ita_deg):
        return "Unknown"
    for label, low, high in ITA_BINS:
        if low < ita_deg <= high:
            return label
    return "Dark (VI)"


def rgb_to_lab_float(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB image (H, W, 3) to float32 CIE L*a*b* (L in [0,100], a,b in [-128,127])."""
    lab_u8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = np.empty_like(lab_u8)
    lab[..., 0] = lab_u8[..., 0] * (100.0 / 255.0)
    lab[..., 1] = lab_u8[..., 1] - 128.0
    lab[..., 2] = lab_u8[..., 2] - 128.0
    return lab


def _extract_background_skin_mask(rgb: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Mask out dermoscope black vignettes, glare, and the central dark lesion.

    Strategy:
    1. Exclude near-black vignette corners (L* < 12 or max(RGB) < 20) and glare (L* > 96).
    2. Among remaining valid pixels, separate lesion vs. surrounding healthy skin
       using Otsu thresholding on L* inside the central region, keeping the lighter
       (non-lesion) skin pixels. Falls back to the outer 25% border ring if Otsu
       leaves too few pixels.
    """
    h, w = rgb.shape[:2]
    l_star = lab[..., 0]

    # Step 1: exclude black dermoscope frame & specular glare
    valid = (l_star >= 12.0) & (l_star <= 96.0) & (rgb.max(axis=-1) >= 20)
    if valid.sum() < 100:
        return valid

    # Step 2: Otsu threshold on L* of valid pixels to separate dark lesion from skin
    valid_l_u8 = np.clip(l_star[valid] * 2.55, 0, 255).astype(np.uint8)
    otsu_thresh_u8, _ = cv2.threshold(
        valid_l_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    otsu_l = otsu_thresh_u8 / 2.55

    # Prefer outer border ring + pixels above Otsu threshold (healthy surrounding skin)
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    dist_norm = np.sqrt(((yy - cy) / (h / 2.0)) ** 2 + ((xx - cx) / (w / 2.0)) ** 2)
    outer_ring = dist_norm >= 0.65

    skin_mask = valid & ((l_star >= otsu_l) | outer_ring)

    # Fallback if mask is too small
    if skin_mask.sum() < 0.05 * h * w:
        skin_mask = valid & (l_star >= np.percentile(l_star[valid], 50))
    if skin_mask.sum() < 50:
        skin_mask = valid

    return skin_mask


def compute_image_ita(image_path: str | Path) -> ITAResult:
    """Compute median non-lesion ITA for a single skin lesion image."""
    path = Path(image_path)
    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)

    lab = rgb_to_lab_float(rgb)
    mask = _extract_background_skin_mask(rgb, lab)

    if mask.sum() == 0:
        return ITAResult(
            filename=path.name,
            ita_deg=float("nan"),
            skin_tone_group="Unknown",
            l_star_median=float("nan"),
            b_star_median=float("nan"),
            skin_pixel_ratio=0.0,
        )

    l_vals = lab[..., 0][mask]
    b_vals = lab[..., 2][mask]

    l_med = float(np.median(l_vals))
    b_med = float(np.median(b_vals))

    # Guard against b* == 0
    if abs(b_med) < 1e-6:
        b_med = 1e-6 if b_med >= 0 else -1e-6

    ita_deg = math.degrees(math.atan((l_med - 50.0) / b_med))

    return ITAResult(
        filename=path.name,
        ita_deg=round(ita_deg, 2),
        skin_tone_group=classify_ita(ita_deg),
        l_star_median=round(l_med, 2),
        b_star_median=round(b_med, 2),
        skin_pixel_ratio=round(float(mask.mean()), 4),
    )


def analyze_image_folder(
    folder_path: str | Path,
    image_paths: Iterable[Path] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Compute ITA scores and summary statistics for all images in a folder."""
    folder = Path(folder_path)
    if image_paths is None:
        paths = sorted(
            p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
    else:
        paths = sorted(Path(p) for p in image_paths)

    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")

    records = [asdict(compute_image_ita(p)) for p in paths]
    df = pd.DataFrame(records)

    valid_ita = df["ita_deg"].dropna()
    group_order = [b[0] for b in ITA_BINS]
    counts = df["skin_tone_group"].value_counts().to_dict()
    total = len(df)

    distribution = {
        group: {
            "count": int(counts.get(group, 0)),
            "percentage": round(100.0 * counts.get(group, 0) / total, 2),
        }
        for group in group_order
    }

    dark_iv_vi = int((valid_ita <= 28.0).sum())
    dark_v_vi = int((valid_ita <= 10.0).sum())

    summary = {
        "total_images": total,
        "valid_images": int(len(valid_ita)),
        "ita_stats": {
            "mean": round(float(valid_ita.mean()), 2),
            "median": round(float(valid_ita.median()), 2),
            "std": round(float(valid_ita.std()), 2),
            "min": round(float(valid_ita.min()), 2),
            "max": round(float(valid_ita.max()), 2),
        },
        "underrepresented_summary": {
            "tan_to_dark_ita_le_28_count": dark_iv_vi,
            "tan_to_dark_ita_le_28_pct": round(100.0 * dark_iv_vi / total, 2),
            "brown_to_dark_ita_le_10_count": dark_v_vi,
            "brown_to_dark_ita_le_10_pct": round(100.0 * dark_v_vi / total, 2),
        },
        "distribution": distribution,
    }
    return df, summary


def save_ita_report(
    df: pd.DataFrame,
    summary: dict,
    output_dir: str | Path,
    prefix: str = "ita",
) -> dict[str, Path]:
    """Save CSV scores and JSON summary to output_dir."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path = out / f"{prefix}_scores.csv"
    json_path = out / f"{prefix}_summary.json"

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {"csv": csv_path, "json": json_path}
