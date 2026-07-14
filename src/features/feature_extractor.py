"""
Extractor de características handcrafted (LBP + HSV).

Combina los descriptores de textura (LBP) y de color (HSV, histograma o energía)
en un único vector por imagen, y recorre un DataFrame para producir la matriz de
características X y el vector de etiquetas y usados por el modelo clásico y por la
rama handcrafted del modelo híbrido.
"""

from src.features.hsv import HSV_extractor
from src.features.lbp import LocalBinaryPattern
import numpy as np
import pandas as pd
from src.utils.constants import LABEL_COL
from src.utils.io import load_image, load_mask


class FeatureExtractor:

    def __init__(self, hsv: HSV_extractor, lbp: LocalBinaryPattern):

        self.hsv_extractor = hsv
        self.lbp_extractor = lbp

    def extract_image_features(self, image: np.ndarray, mask: np.ndarray, use_energies: bool) -> np.ndarray:
        if use_energies:
            hsv_features = self.hsv_extractor.compute_energy_histograms(image, mask)
        else:

            hsv_features = self.hsv_extractor.extract_hsv_features(image, mask)
        lbp_features = self.lbp_extractor.extract_lbp_features(image, mask)

        # Vector final: LBP+HSV
        return np.concatenate([lbp_features, hsv_features])

    def extract_features(self, dataset: pd.DataFrame, image_column_name: str, mask_column_name: str,
                          use_energies: bool = False) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for _, row in dataset.iterrows():
            # imagen en formato RGB

            image = load_image(row[image_column_name])
            # mascara en escala de gris
            mask = load_mask(row[mask_column_name])

            X.append(self.extract_image_features(image, mask, use_energies))
            y.append(row[LABEL_COL])
        return np.array(X), np.array(y)