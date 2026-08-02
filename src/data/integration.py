from pathlib import Path
import pandas as pd
import json

class DatasetIntegrator:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)

        self.output_path = Path(
            self.config["paths"]["integrated"]
        )

        self.output_path.mkdir(
            parents=True,
            exist_ok=True
        )

    def _load_config(self, config_path):
        import yaml

        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def _filter_isic_melanoma(self):
        isic_path = Path(
            self.config["datasets"]["isic"]["metadata"]
        )

        df = pd.read_csv(isic_path)
            
        melanoma = df[
            df["diagnosis"]
            .str.lower()
            .str.contains("melanoma", na=False)
        ].copy()

        melanoma["image_id"] = melanoma["image_name"]

        melanoma["image_path"] = (
            self.config["datasets"]["isic"]["image_dir"]
            + "/"
            + melanoma["image_name"]
            + ".jpg"
        )

        melanoma["fitzpatrick_label"] = None

        return melanoma[
            [
                "image_id",
                "image_path",
                "fitzpatrick_label"
            ]
        ], len(df)

    def _filter_fitzpatrick_melanoma(self):
        fitz_path = Path(
            self.config["datasets"]["fitzpatrick"]["metadata"]
        )

        df = pd.read_csv(fitz_path)

        melanoma = df[
            df["label"]
            .str.lower()
            .str.contains("melanoma", na=False)
        ].copy()

        melanoma["image_id"] = melanoma["md5hash"]

        melanoma["image_path"] = (
            self.config["datasets"]["fitzpatrick"]["image_dir"]
            + "/" 
            + melanoma["md5hash"] 
            + ".jpg"
        )

        melanoma["fitzpatrick_label"] = (
            melanoma["fitzpatrick_scale"]
        )

        return melanoma[
            [
                "image_id",
                "image_path",
                "fitzpatrick_label"
            ]
        ], len(df)

    def _generate_report(self, isic_melanoma, isic, fitz_melanoma, fitz): 
        report = {
            "isic": {
                "original_samples": isic,
                "melanoma_samples": isic_melanoma
            },
            "fitzpatrick17k": {
                "original_samples": fitz,
                "melanoma_samples": fitz_melanoma
            },
            "integrated_dataset": {
                "total_samples": fitz_melanoma + isic_melanoma
            }
        }

        Path("reports/integration").mkdir(
            parents=True,
            exist_ok=True
        )

        path = "reports/integration/integration_report.json"

        with open(
            path,
            "w"
        ) as file:
            json.dump(report, file, indent=4)

        print(f"Report saved as: {path}")

    def run(self):
        print("Integrating datasets...")

        isic, isic_count = self._filter_isic_melanoma()
        fitzpatrick, fitz_count = self._filter_fitzpatrick_melanoma()

        integrated = pd.concat(
            [
                isic,
                fitzpatrick
            ],
            ignore_index=True
        )

        output_file = (
            self.output_path /
            "metadata.csv"
        )

        integrated.to_csv(
            output_file,
            index=False
        )

        print(f"ISIC melanoma: {len(isic)} of total {isic_count}")
        print(f"Fitzpatrick melanoma: {len(fitzpatrick)} of total {fitz_count}")
        print(f"Total images: {len(integrated)}")
        print(f"Saved: {output_file}")

        self._generate_report(len(isic), isic_count, len(fitzpatrick), fitz_count)

if __name__ == "__main__":
    integrator = DatasetIntegrator(
        "configs/config.yaml"
    )

    integrator.run()