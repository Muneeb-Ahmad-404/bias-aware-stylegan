import shutil
from pathlib import Path

import kagglehub
import yaml


class DatasetIngestor:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.raw_path = Path(self.config["paths"]["raw"])

        self.raw_path.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def download_dataset(self, dataset_name, dataset_config):
        print(f"Downloading {dataset_name}")

        downloaded_path = Path(kagglehub.dataset_download(dataset_config["kaggle"]))

        destination = self.raw_path / dataset_config["name"]

        if destination.exists():
            print(f"{destination} already exists")
            return

        shutil.copytree(downloaded_path, destination, dirs_exist_ok=True)

        print(f"Saved to {destination}")

    def run(self):
        for name, dataset in self.config["datasets"].items():
            self.download_dataset(name, dataset)


if __name__ == "__main__":
    ingestor = DatasetIngestor("configs/config.yaml")

    ingestor.run()
