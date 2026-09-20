from pathlib import Path
import yaml
import logging as logger
import torchvision.models as models
import torch
import torch.nn as nn
import cv2 as cv
from cv2.typing import MatLike
import numpy as np
import os

class ItaInference:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.path = Path(self.config["datasets"]["isic"]["image_dir"])
        logger.basicConfig(level=logger.INFO)
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
        images = os.listdir(self.path)
        logger.info(f"Found {len(images)} images to process.")

        # for testing
        # test_images = [
        #     'ISIC_9922133.jpg',
        #     'ISIC_9922430.jpg',
        #     'ISIC_9923018.jpg',
        #     'ISIC_9886540.jpg',
        #     'ISIC_9991451.jpg',
        #     'ISIC_0068279.jpg',
        # ]

        for img_name in images:
            img_path = self.path / img_name
            if not img_path.exists():
                logger.warning(f"Image not found: {img_path}")
                continue
                
            img_bgr = cv.imread(str(img_path))

            if img_bgr is not None:
                yield img_name, img_bgr

    def remove_hair_multiscale(self, img_bgr: MatLike) -> MatLike:
        h, w = img_bgr.shape[:2]
        
        # Downscale for faster inpainting
        scale_factor = 800.0 / max(w, h)
        if scale_factor < 1.0:
            small_img = cv.resize(img_bgr, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_AREA)
        else:
            small_img = img_bgr.copy()
            
        small_gray = cv.cvtColor(small_img, cv.COLOR_BGR2GRAY)
        accum_mask = np.zeros_like(small_gray, dtype=np.uint8)

        # 1. Multiscale Blackhat to catch different hair diameters
        kernel_sizes = [15, 25, 35]
        for k_size in kernel_sizes:
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (k_size, k_size))
            khat = cv.morphologyEx(small_gray, cv.MORPH_BLACKHAT, kernel)
            
            # Normalize and threshold dynamically
            khat_norm = cv.normalize(khat, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
            min_val, max_val, _, _ = cv.minMaxLoc(khat_norm)
            thresh_val = min_val + 0.15 * (max_val - min_val)
            
            _, msk = cv.threshold(khat_norm, thresh_val, 255, cv.THRESH_BINARY)
            
            # Dilate to ensure the mask covers the edges of the hair shaft
            dilate_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
            msk = cv.dilate(msk, dilate_kernel)
            
            accum_mask = cv.bitwise_or(accum_mask, msk)

        # 2. Check if hair was actually found
        if cv.countNonZero(accum_mask) == 0:
            return img_bgr.copy()
            
        # 3. Inpaint the hair pixels
        hair_free_small = cv.inpaint(small_img, accum_mask, 3.0, cv.INPAINT_TELEA)
        
        # 4. Upscale back to original resolution if it was downscaled
        if scale_factor < 1.0:
            hair_free_img = cv.resize(hair_free_small, (w, h), interpolation=cv.INTER_CUBIC)
            return hair_free_img
        return hair_free_small
    
    def get_prediction(self):
        pass

    def calculate_ita(self):
        pass

    def calculate_fitzscale(self):
        pass

    def lesion_mask(self): 
        pass

    def run_inference(self):
        for img_name, img_bgr in self.load_images():
            logger.info(f"Removing hair for {img_name}...")
            hair_free_img = self.remove_hair_multiscale(img_bgr)

            # Debug: Save it temporarily to verify
            # cv.imwrite(f"debug_{img_name}", hair_free_img)
        
if __name__ == '__main__':
    ita = ItaInference('configs/config.yaml')
    ita.run_inference()