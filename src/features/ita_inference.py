from pathlib import Path
import yaml
import logging as logger
import torchvision.models as models
import torch
import torch.nn as nn

class ItaInference:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.path = Path(self.config["Datasets"]["isic"]["image_dir"])
        if not self.path.exists():
            logger.error('The directory does not exist')
            raise Exception('The directory does not exist')
            
        self.models = []
        
    def _load_config(self, config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def load_models(self):
        base_dir = Path('finetune_mskcc_blur0_lab')
        if not base_dir.exists():
            logger.error('The directory does not exist')
            raise Exception('The directory does not exist')
            
        fold_models = []

        for fold in range(5):
            model = models.efficientnet_b4(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, 3)
            
            weights_path = base_dir / f'fold_{fold}' / 'model.pth'
            state_dict = torch.load(weights_path, map_location='cpu')
            
            model.load_state_dict(state_dict)
            model.eval()
            
            fold_models.append(model)

        logger.info(f"Successfully loaded {len(fold_models)} EfficientNet-B4 LAB models.")

        self.models = fold_models

    def load_images(self):
        pass

    def get_prediction(self):
        pass

    def calculate_ita(self):
        pass

    def calculate_fitzscale(self):
        pass

    def hair_mask(self):
        pass

    def lesion_mask(self):
        pass

    def dull_raisor(self):
        pass


if __name__ == '__main__':
    ita = ItaInference('configs/config.yaml')
    ita.load_models()