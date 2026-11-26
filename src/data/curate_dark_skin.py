"""Curate darker-skin melanoma subsets using colorimetric ITA estimation.

Reads `data/integrated/metadata.csv`, computes CIE L*a*b* ITA scores for all
available melanoma images (ISIC 2020 dermoscopic + Fitzpatrick17k clinical),
writes annotated metadata + summary statistics, and exports curated 512x512
image folders for StyleGAN2-ADA fine-tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

from src.evaluation.ita import IMAGE_EXTENSIONS, ITA_BINS, compute_image_ita


def _build_raw_image_index(raw_dir: Path = Path("data/raw")) -> dict[str, Path]:
    """Index image files under data/raw by stem so mismatched metadata paths still resolve."""
    index: dict[str, Path] = {}
    if not raw_dir.exists():
        return index
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(p.stem, p)
    return index


def _resolve_image_path(row: pd.Series, raw_index: dict[str, Path]) -> Path | None:
    p = Path(str(row["image_path"]))
    if p.exists():
        return p
    for key in (p.stem, str(row["image_id"])):
        if key in raw_index:
            return raw_index[key]
    return None


def _infer_source(image_path: str) -> str:
    lower = image_path.lower()
    if "isic" in lower:
        return "isic2020"
    if "fitzpatrick" in lower:
        return "fitzpatrick17k"
    return "unknown"


def _summarize_group(df: pd.DataFrame) -> dict:
    valid = df["ita_deg"].dropna()
    total = len(df)
    counts = df["skin_tone_group"].value_counts().to_dict()
    group_order = [b[0] for b in ITA_BINS]

    return {
        "total_images": int(total),
        "valid_images": int(len(valid)),
        "ita_stats": {
            "mean": round(float(valid.mean()), 2) if len(valid) else None,
            "median": round(float(valid.median()), 2) if len(valid) else None,
            "std": round(float(valid.std()), 2) if len(valid) else None,
            "min": round(float(valid.min()), 2) if len(valid) else None,
            "max": round(float(valid.max()), 2) if len(valid) else None,
        },
        "thresholds": {
            "ita_le_41_intermediate_to_dark": int((valid <= 41.0).sum()),
            "ita_le_28_tan_to_dark": int((valid <= 28.0).sum()),
            "ita_le_10_brown_to_dark": int((valid <= 10.0).sum()),
            "ita_le_neg30_dark": int((valid <= -30.0).sum()),
        },
        "distribution": {
            g: {
                "count": int(counts.get(g, 0)),
                "percentage": round(100.0 * counts.get(g, 0) / total, 2) if total else 0.0,
            }
            for g in group_order
        },
    }


ITA_BIN_TO_ID = {name: idx for idx, (name, _, _) in enumerate(ITA_BINS)}


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _export_subset(
    df: pd.DataFrame,
    dest_dir: Path,
    size: int = 512,
    write_dataset_json: bool = False,
) -> int:
    """Copy (and normalize to RGB 512x512 PNG) selected images into dest_dir."""
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    exported = 0
    labels: list[list[str | int]] = []
    for _, row in df.iterrows():
        src = Path(row["image_path"])
        if not src.exists():
            continue
        out_name = f"{row['source']}_{row['image_id']}.png"
        out_path = dest_dir / out_name
        with Image.open(src) as img:
            rgb = img.convert("RGB")
            if rgb.size != (size, size):
                rgb = rgb.resize((size, size), Image.Resampling.LANCZOS)
            rgb.save(out_path, format="PNG")
        if write_dataset_json and pd.notna(row.get("ita_bin_id")):
            labels.append([out_name, int(row["ita_bin_id"])])
        exported += 1

    if write_dataset_json:
        with open(dest_dir / "dataset.json", "w", encoding="utf-8") as f:
            json.dump({"labels": labels}, f, indent=2)
    return exported


def curate_dark_skin(
    metadata_csv: str | Path = "data/integrated/metadata.csv",
    processed_dir: str | Path = "data/processed",
    report_dir: str | Path = "reports/curation",
    ita_threshold: float = 28.0,
) -> dict:
    meta_path = Path(metadata_csv)
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    df_meta = pd.read_csv(meta_path)
    raw_index = _build_raw_image_index()
    seen_hashes: set[str] = set()
    records = []
    missing_count = 0
    duplicates_removed = 0

    for _, row in df_meta.iterrows():
        img_path = _resolve_image_path(row, raw_index)
        if img_path is None:
            missing_count += 1
            continue

        digest = _file_sha256(img_path)
        if digest in seen_hashes:
            duplicates_removed += 1
            continue
        seen_hashes.add(digest)

        ita_res = compute_image_ita(img_path)
        fitz_val = row.get("fitzpatrick_label")
        records.append(
            {
                "image_id": row["image_id"],
                "source": _infer_source(str(row["image_path"])),
                "image_path": str(img_path),
                "fitzpatrick_label": None if pd.isna(fitz_val) else int(fitz_val),
                "ita_deg": ita_res.ita_deg,
                "skin_tone_group": ita_res.skin_tone_group,
                "ita_bin_id": ITA_BIN_TO_ID.get(ita_res.skin_tone_group),
                "l_star_median": ita_res.l_star_median,
                "b_star_median": ita_res.b_star_median,
                "skin_pixel_ratio": ita_res.skin_pixel_ratio,
            }
        )

    if not records:
        raise FileNotFoundError(
            "No images found on disk. Run `python src/data/ingest.py` first."
        )

    df_ita = pd.DataFrame(records)

    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)
    annotated_csv = proc_path / "melanoma_ita_metadata.csv"
    df_ita.to_csv(annotated_csv, index=False)

    # Subset 1: Pure dermoscopic (ISIC 2020) with ITA <= ita_threshold
    isic_df = df_ita[df_ita["source"] == "isic2020"]
    isic_dark = isic_df[isic_df["ita_deg"] <= ita_threshold]

    # Subset 2: Combined (ISIC ITA <= threshold OR Fitzpatrick ITA <= threshold OR FST 4..6)
    fitz_df = df_ita[df_ita["source"] == "fitzpatrick17k"]
    fitz_dark = fitz_df[
        (fitz_df["ita_deg"] <= ita_threshold)
        | (fitz_df["fitzpatrick_label"].isin([4, 5, 6]))
    ]
    combined_dark = pd.concat([isic_dark, fitz_dark], ignore_index=True)

    tag = str(int(ita_threshold))
    isic_dir = proc_path / f"finetune_isic_ita{tag}"
    fitz_dir = proc_path / f"finetune_fitzpatrick_ita{tag}"
    combined_dir = proc_path / f"finetune_combined_ita{tag}"
    isic_cond_dir = proc_path / "isic_ita_conditioned"

    n_isic_exported = _export_subset(isic_dark, isic_dir)
    n_fitz_exported = _export_subset(fitz_dark, fitz_dir)
    n_combined_exported = _export_subset(combined_dark, combined_dir)
    n_isic_cond_exported = _export_subset(
        isic_df.dropna(subset=["ita_bin_id"]), isic_cond_dir, write_dataset_json=True
    )

    summary = {
        "missing_images": int(missing_count),
        "duplicates_removed": int(duplicates_removed),
        "ita_threshold_deg": ita_threshold,
        "ita_bin_mapping": ITA_BIN_TO_ID,
        "all_melanoma": _summarize_group(df_ita),
        "isic2020_melanoma": _summarize_group(isic_df),
        "fitzpatrick17k_melanoma": _summarize_group(fitz_df),
        "exported_subsets": {
            "isic_only": {
                "dir": str(isic_dir),
                "count": int(n_isic_exported),
            },
            "fitzpatrick_only": {
                "dir": str(fitz_dir),
                "count": int(n_fitz_exported),
            },
            "combined": {
                "dir": str(combined_dir),
                "count": int(n_combined_exported),
            },
            "isic_ita_conditioned": {
                "dir": str(isic_cond_dir),
                "count": int(n_isic_cond_exported),
            },
        },
    }

    rep_path = Path(report_dir)
    rep_path.mkdir(parents=True, exist_ok=True)
    summary_json = rep_path / "real_melanoma_ita_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate dark-skin melanoma subsets via ITA.")
    parser.add_argument("--metadata", default="data/integrated/metadata.csv")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--report-dir", default="reports/curation")
    parser.add_argument("--ita-threshold", type=float, default=28.0)
    args = parser.parse_args()

    summary = curate_dark_skin(
        metadata_csv=args.metadata,
        processed_dir=args.processed_dir,
        report_dir=args.report_dir,
        ita_threshold=args.ita_threshold,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
