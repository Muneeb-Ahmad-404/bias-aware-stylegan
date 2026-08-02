import json
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image


class DatasetValidator:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg"}

    def __init__(self, config_path: str):
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)

        self.raw_path = Path(self.config["paths"]["raw"])

        self.report_path = Path("reports/validation")
        self.report_path.mkdir(parents=True, exist_ok=True)

        self.results = {}

    def validate_image_folder(self, folder_path: Path):
        images = []

        for ext in self.IMAGE_EXTENSIONS:
            images.extend(folder_path.glob(f"*{ext}"))

        corrupted = 0
        resolutions = {}

        for image_path in images:
            try:
                with Image.open(image_path) as image:
                    image.verify()

                with Image.open(image_path) as image:
                    resolution = f"{image.width}x{image.height}"

                resolutions[resolution] = resolutions.get(resolution, 0) + 1

            except Exception:
                corrupted += 1

        return {
            "image_count": len(images),
            "corrupted_images": corrupted,
            "resolutions": resolutions,
            "image_paths": images,
        }

    def validate_csv(self, csv_path: Path):
        if not csv_path.exists():
            return {"exists": False, "rows": 0, "dataframe": None}

        dataframe = pd.read_csv(csv_path)

        return {"exists": True, "rows": len(dataframe), "dataframe": dataframe}

    def validate_metadata_mapping(self, image_paths, dataframe, image_column):
        if dataframe is None:
            return {"mapping_valid": False, "reason": "CSV not found"}

        if image_column not in dataframe.columns:
            return {"mapping_valid": False, "reason": f"Missing column: {image_column}"}

        image_ids = {image.stem for image in image_paths}

        csv_ids = {str(value).replace(".jpg", "") for value in dataframe[image_column]}

        return {
            "images_without_metadata": list(image_ids - csv_ids),
            "metadata_without_images": list(csv_ids - image_ids),
            "mapping_valid": image_ids == csv_ids,
        }

    def validate_isic(self):
        dataset_path = self.raw_path / self.config["datasets"]["isic"]["name"]

        train_images = self.validate_image_folder(dataset_path / "train")

        test_images = self.validate_image_folder(dataset_path / "test")

        train_csv = self.validate_csv(dataset_path / "train.csv")

        test_csv = self.validate_csv(dataset_path / "test.csv")

        self.results["isic"] = {
            "train": {
                "image_count": train_images["image_count"],
                "corrupted_images": train_images["corrupted_images"],
                "resolutions": train_images["resolutions"],
                "csv_rows": train_csv["rows"],
                "count_matches_csv": train_images["image_count"] == train_csv["rows"],
                "metadata_mapping": self.validate_metadata_mapping(
                    train_images["image_paths"], train_csv["dataframe"], "image_name"
                ),
            },
            "test": {
                "image_count": test_images["image_count"],
                "corrupted_images": test_images["corrupted_images"],
                "resolutions": test_images["resolutions"],
                "csv_rows": test_csv["rows"],
                "count_matches_csv": test_images["image_count"] == test_csv["rows"],
                "metadata_mapping": self.validate_metadata_mapping(
                    test_images["image_paths"], test_csv["dataframe"], "image"
                ),
            },
        }

    def validate_fitzpatrick(self):
        dataset_path = self.raw_path / self.config["datasets"]["fitzpatrick"]["name"]

        images = self.validate_image_folder(dataset_path / "data")

        csv = self.validate_csv(dataset_path / "fitzpatrick17k.csv")

        self.results["fitzpatrick"] = {
            "image_count": images["image_count"],
            "corrupted_images": images["corrupted_images"],
            "resolutions": images["resolutions"],
            "csv_rows": csv["rows"],
            "count_matches_csv": images["image_count"] == csv["rows"],
            "metadata_mapping": self.validate_metadata_mapping(
                images["image_paths"], csv["dataframe"], "md5hash"
            ),
        }

    def validate(self):
        self.validate_isic()
        self.validate_fitzpatrick()

    def save_report(self):
        report_file = self.report_path / "validation.json"

        with open(report_file, "w") as file:
            json.dump(self.results, file, indent=4)

        print(f"Validation report saved: {report_file}")


def main():
    validator = DatasetValidator("configs/config.yaml")

    validator.validate()
    validator.save_report()


if __name__ == "__main__":
    main()
