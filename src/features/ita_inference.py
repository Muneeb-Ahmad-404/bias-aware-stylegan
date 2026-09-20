import logging as logger
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import yaml
from cv2.typing import MatLike


class ItaInference:
    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self.path = Path(self.config["datasets"]["isic"]["image_dir"])
        logger.basicConfig(level=logger.INFO)
        if not self.path.exists():
            logger.error("The directory does not exist")
            raise Exception("The directory does not exist")

        self.models = []

    def _load_config(self, config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    def load_models(self):
        base_dir = Path("finetune_mskcc_blur0_lab")
        if not base_dir.exists():
            logger.error("The directory does not exist")
            raise Exception("The directory does not exist")

        fold_models = []

        for fold in range(5):
            model = models.efficientnet_b4(weights=None)
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, 3)

            weights_path = base_dir / f"fold_{fold}" / "model.pth"
            state_dict = torch.load(weights_path, map_location="cpu")

            model.load_state_dict(state_dict)
            model.eval()

            fold_models.append(model)

        logger.info(
            f"Successfully loaded {len(fold_models)} EfficientNet-B4 LAB models."
        )

        self.models = fold_models

    def load_images(self):
        images = os.listdir(self.path)
        logger.info(f"Found {len(images)} images to process.")

        # for testing
        test_images = [
            "ISIC_9922133.jpg",
            "ISIC_9922430.jpg",
            "ISIC_9923018.jpg",
            "ISIC_9886540.jpg",
            "ISIC_9991451.jpg",
            "ISIC_0068279.jpg",
        ]

        for img_name in test_images:
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
            small_img = cv.resize(
                img_bgr,
                (0, 0),
                fx=scale_factor,
                fy=scale_factor,
                interpolation=cv.INTER_AREA,
            )
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
            hair_free_img = cv.resize(
                hair_free_small, (w, h), interpolation=cv.INTER_CUBIC
            )
            return hair_free_img
        return hair_free_small

    def extract_clean_patch(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (5, 5), 0)

        # Detect dark regions
        _, mask = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        # Clean mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

        h, w = mask.shape

        # Remove components touching image borders
        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(
            mask, connectivity=8
        )

        candidates = []

        for label in range(1, num_labels):
            x, y, bw, bh, area = stats[label]

            touches_border = x == 0 or y == 0 or x + bw >= w or y + bh >= h

            if not touches_border:
                candidates.append((area, label, x, y, bw, bh))

        if not candidates:
            return None

        # Largest non-border dark component = lesion candidate
        area, label, x, y, bw, bh = max(candidates, key=lambda item: item[0])

        lesion_mask = labels == label

        # find skin around the lision
        lesion_center_x = x + bw // 2
        lesion_center_y = y + bh // 2

        # Candidate patch sizes in original 512x512 image
        patch_sizes = [128, 96, 64]

        # Search positions around lesion
        offsets = [
            (-1, -1),
            (0, -1),
            (1, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
        ]

        for patch_size in patch_sizes:
            half = patch_size // 2

            for dx, dy in offsets:
                cx = lesion_center_x + dx * (bw // 2 + patch_size)
                cy = lesion_center_y + dy * (bh // 2 + patch_size)

                x1 = cx - half
                y1 = cy - half
                x2 = x1 + patch_size
                y2 = y1 + patch_size

                # Must stay inside image
                if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                    continue

                local_lesion = lesion_mask[y1:y2, x1:x2]

                # Reject patch if it contains lesion
                lesion_fraction = np.mean(local_lesion)

                if lesion_fraction > 0.05:
                    continue

                patch = img[y1:y2, x1:x2]

                return patch

        return None

    def get_prediction(self, img_bgr):
        # preprocess to pil
        img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        resized_image = cv.resize(img_rgb, (224, 224))
        normalized_image = resized_image / 255.0
        input_tensor = torch.tensor(normalized_image, dtype=torch.float32).permute(
            2, 0, 1
        )
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        input_tensor = normalize(input_tensor).unsqueeze(0)

        outputs_list = []

        with torch.no_grad():
            for model in self.models:
                pred = model(input_tensor)
                outputs_list.append(pred)

        final_prediction = torch.stack(outputs_list).mean(dim=0)
        return (
            final_prediction[0][0].item(),
            final_prediction[0][1].item(),
            final_prediction[0][2].item(),
        )
        # return l, a, b

    def calculate_ita(self, l, b):
        # Avoid division by zero in extreme edge cases
        if b == 0:
            b = 0.0001

        # Calculate the angle in radians
        ratio = (l - 50) / b
        ita_radians = math.atan(ratio)

        return ita_radians * (180 / math.pi)

    def calculate_fitzscale(self, ita):
        ita_bnd_kin = -1

        if ita > 55:
            ita_bnd_kin = 1
        if 41 < ita <= 55:
            ita_bnd_kin = 2
        if 28 < ita <= 41:
            ita_bnd_kin = 3
        if 19 < ita <= 28:
            ita_bnd_kin = 4
        if 10 < ita <= 19:
            ita_bnd_kin = 5
        if ita <= 10:
            ita_bnd_kin = 6

        return ita_bnd_kin

    def run_inference(self):
        results = []

        for img_name, img_bgr in self.load_images():
            logger.info(f"Processing {img_name}...")

            hair_free_img = self.remove_hair_multiscale(img_bgr)
            clean_patch = self.extract_clean_patch(hair_free_img)

            if clean_patch is None:
                logger.warning(f"{img_name}: could not extract clean patch")
                continue

            l, a, b = self.get_prediction(clean_patch)
            ita_val = self.calculate_ita(l, b)
            fitz_val = self.calculate_fitzscale(ita_val)

            results.append(
                {
                    "image": img_name,
                    "l": l,
                    "a": a,
                    "b": b,
                    "ita": ita_val,
                    "fitzpatrick": fitz_val,
                }
            )

            logger.info(
                f"{img_name}: L={l:.2f}, a={a:.2f}, "
                f"b={b:.2f}, ITA={ita_val:.2f}, Fitz={fitz_val}"
            )

            # Debug: Save it temporarily to verify
            cv.imwrite(f"hair_free_{img_name}", hair_free_img)
            cv.imwrite(f"skin_patch_{img_name}", clean_patch)

        # self.log_experiment(
        #     results,
        #     experiment_id="003",
        #     title="Lesion-Aware Skin Patch",
        #     objective=(
        #         "Evaluate whether estimating skin tone from a lesion-free "
        #         "surrounding-skin patch reduces the influence of melanoma lesions "
        #         "on skin-tone estimation."
        #     ),
        #     change_from_previous=(
        #         "A lesion-aware surrounding-skin crop was introduced after the "
        #         "multiscale hair-removal step. The lesion is detected using Otsu "
        #         "thresholding and border-connected components are excluded. A "
        #         "contiguous lesion-free skin patch surrounding the detected lesion "
        #         "is then selected for model inference. All other inference steps "
        #         "remain unchanged."
        #     ),
        #     preprocessing=[
        #         "BGR → RGB",
        #         "Multiscale blackhat hair detection",
        #         "Hair mask dilation",
        #         "Telea inpainting",
        #         "Otsu thresholding for lesion detection",
        #         "Removal of border-connected mask components",
        #         "Selection of a lesion-free surrounding-skin patch",
        #         "Resize selected patch to 224×224",
        #         "Convert pixel values to [0, 1]",
        #         "ImageNet normalization",
        #     ],
        # )

    def log_experiment(
        self,
        results,
        experiment_id,
        title,
        objective,
        preprocessing,
        change_from_previous,
        observations="TODO",
        next_experiment="TODO",
    ):
        log_dir = Path("reports/experiments/ita")
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"{experiment_id}_{title.lower().replace(' ', '_')}.md"

        with open(log_file, "w") as f:
            f.write(f"# Experiment {experiment_id} — {title}\n\n")

            f.write("## Objective\n\n")
            f.write(f"{objective}\n\n")

            f.write("## Date\n\n")
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Input\n\n")
            f.write("- Dataset: ISIC 2020\n")
            f.write("- Subset: current test images\n")
            f.write(f"- Images processed: {len(results)}\n")
            f.write(f"- Image directory: `{self.path}`\n")
            f.write("- Original resolution: 512×512\n\n")

            f.write("## Change from Previous Experiment\n\n")
            f.write(f"{change_from_previous}\n\n")

            f.write("## Preprocessing\n\n")
            for step in preprocessing:
                f.write(f"- {step}\n")
            f.write("\n")

            f.write("## Model\n\n")
            f.write("- Architecture: EfficientNet-B4\n")
            f.write("- Task: Lab regression\n")
            f.write("- Output: L*, a*, b*\n")
            f.write("- Number of folds: 5\n")
            f.write("- Aggregation: mean L*, a*, b* across folds\n\n")

            f.write("## ITA Calculation\n\n")
            f.write("ITA = atan((L - 50) / b) × 180 / π\n\n")

            f.write("## Fitzpatrick Mapping\n\n")
            f.write("- ITA > 55 → I\n")
            f.write("- 41 < ITA ≤ 55 → II\n")
            f.write("- 28 < ITA ≤ 41 → III\n")
            f.write("- 19 < ITA ≤ 28 → IV\n")
            f.write("- 10 < ITA ≤ 19 → V\n")
            f.write("- ITA ≤ 10 → VI\n\n")

            f.write("## Results\n\n")
            f.write("| Image | L* | a* | b* | ITA | Fitzpatrick |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")

            for result in results:
                f.write(
                    f"| {result['image']} | "
                    f"{result['l']:.3f} | "
                    f"{result['a']:.3f} | "
                    f"{result['b']:.3f} | "
                    f"{result['ita']:.3f} | "
                    f"{result['fitzpatrick']} |\n"
                )

            f.write("\n## Observations\n\n")
            f.write(f"{observations}\n\n")

            f.write("## Decision / Next Experiment\n\n")
            f.write(f"{next_experiment}\n")


if __name__ == "__main__":
    ita = ItaInference("configs/config.yaml")
    ita.load_models()
    ita.run_inference()
