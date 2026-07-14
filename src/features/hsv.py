"""
Descriptores de color en espacio HSV.

Ofrece dos familias de características de color sobre la imagen en HSV:
histogramas de color por canal (H, S, V) normalizados, e histogramas de la
energía (módulo del gradiente) de cada canal. Ambos pueden restringirse a la
región de la máscara.
"""

import numpy as np
import cv2

from src.utils.constants import NUMPOINTS


class HSV_extractor:

    def __init__(self, n_bins: int = NUMPOINTS + 2):
        self.n_bins = n_bins

    def extract_hsv_features(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:

        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        features = []
        # Un histograma por canal; los rangos difieren (H: 0-180, S y V: 0-256).
        for channel, r in zip([0, 1, 2], [(0, 180), (0, 256), (0, 256)]):
            hist = cv2.calcHist([img_hsv], [channel], mask, [self.n_bins], list(r))
            cv2.normalize(hist, hist)  # rango [0,1]
            features.append(hist.flatten())

        return np.concatenate(features)

    def compute_color_energy(self, image: np.ndarray) -> list[np.ndarray]:
        # int16 para permitir diferencias con signo sin desbordar (uint8 no basta).
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)

        energy_arrays = []
        for c in range(3):  # H, S, V
            channel = img_hsv[:, :, c]

            # Gradientes horizontal (gx) y vertical (gy) por diferencias centradas.
            gx = channel[1:-1, 2:] - channel[1:-1, :-2]
            gy = channel[2:, 1:-1] - channel[:-2, 1:-1]

            # Energía = módulo del gradiente en cada píxel interior.
            energy = np.sqrt(gx.astype(np.float32) ** 2 + gy.astype(np.float32) ** 2)

            energy_arrays.append(energy.ravel())

        return energy_arrays

    def compute_energy_histograms(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        energy_arrays = self.compute_color_energy(image)

        # Cotas teóricas del módulo de la energía por canal:
        #   H: gx, gy ∈ [-180, 180]  → energy ∈ [0, sqrt(2)·180] ≈ [0, 254.6]
        #   S: gx, gy ∈ [-255, 255]  → energy ∈ [0, sqrt(2)·255] ≈ [0, 360.6]
        #   V: igual que S
        ranges = [(0.0, 255.0), (0.0, 361.0), (0.0, 361.0)]

        features = []
        for energy_vals, rng in zip(energy_arrays, ranges):
            # La máscara se recorta a la zona interior (los gradientes pierden el borde).
            if mask is not None:
                mask_inner = mask[1:-1, 1:-1].astype(bool)
                energy_vals = energy_vals[mask_inner.ravel()]

            hist, _ = np.histogram(energy_vals, bins=self.n_bins,
                                   range=rng, density=True)
            features.append(hist)

        return np.concatenate(features)