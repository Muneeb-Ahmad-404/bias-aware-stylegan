"""Blinded clinical review set exporter (FR12 / NFR4).

Shuffles synthetic (and optional real control) melanoma images, assigns anonymous
case IDs (`case_0001.png`, ...), writes a clinician-facing `review_sheet.csv`
without source/ITA labels, and stores a separate `review_key.csv` for unblinding.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import pandas as pd

from src.evaluation.ita import IMAGE_EXTENSIONS, compute_image_ita


def export_blinded_review_set(
    synthetic_dir: str | Path,
    output_dir: str | Path = "reports/clinical_review",
    n_synthetic: int = 100,
    real_dir: str | Path | None = None,
    n_real: int = 0,
    seed: int = 42,
) -> dict[str, Path | int]:
    """Create a blinded image folder + clinician scoring sheet + hidden key."""
    rng = random.Random(seed)
    syn_paths = sorted(
        p for p in Path(synthetic_dir).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not syn_paths:
        raise FileNotFoundError(f"No images found in {synthetic_dir}")

    chosen: list[tuple[Path, str]] = [
        (p, "synthetic") for p in rng.sample(syn_paths, min(n_synthetic, len(syn_paths)))
    ]

    if real_dir is not None and n_real > 0:
        real_paths = sorted(
            p for p in Path(real_dir).iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        chosen.extend(
            (p, "real") for p in rng.sample(real_paths, min(n_real, len(real_paths)))
        )

    rng.shuffle(chosen)

    out_path = Path(output_dir)
    cases_dir = out_path / "images"
    if cases_dir.exists():
        shutil.rmtree(cases_dir)
    cases_dir.mkdir(parents=True, exist_ok=True)

    key_rows = []
    sheet_rows = []
    for idx, (src, origin) in enumerate(chosen, start=1):
        case_id = f"case_{idx:04d}"
        dest = cases_dir / f"{case_id}{src.suffix.lower()}"
        shutil.copy2(src, dest)
        ita_res = compute_image_ita(src)
        key_rows.append(
            {
                "case_id": case_id,
                "origin": origin,
                "original_file": src.name,
                "ita_deg": ita_res.ita_deg,
                "skin_tone_group": ita_res.skin_tone_group,
            }
        )
        sheet_rows.append(
            {
                "case_id": case_id,
                "image_file": dest.name,
                "realistic_1_to_5": "",
                "melanoma_plausible_yes_no": "",
                "notes": "",
            }
        )

    sheet_path = out_path / "review_sheet.csv"
    key_path = out_path / "review_key.csv"
    pd.DataFrame(sheet_rows).to_csv(sheet_path, index=False)
    pd.DataFrame(key_rows).to_csv(key_path, index=False)

    return {
        "total_cases": len(chosen),
        "images_dir": cases_dir,
        "review_sheet": sheet_path,
        "review_key": key_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export blinded clinical review set.")
    parser.add_argument("--synthetic-dir", required=True)
    parser.add_argument("--output-dir", default="reports/clinical_review")
    parser.add_argument("--n-synthetic", type=int, default=100)
    parser.add_argument("--real-dir", default=None)
    parser.add_argument("--n-real", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    res = export_blinded_review_set(
        synthetic_dir=args.synthetic_dir,
        output_dir=args.output_dir,
        n_synthetic=args.n_synthetic,
        real_dir=args.real_dir,
        n_real=args.n_real,
        seed=args.seed,
    )
    print(res)


if __name__ == "__main__":
    main()
