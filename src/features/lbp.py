"""
Descriptor de textura Local Binary Pattern (LBP).

Calcula el histograma LBP (método 'uniform') sobre el canal de intensidad (V) de
la imagen en espacio HSV, restringido a la región de la máscara si se
proporciona. El histograma normalizado es el vector de características de textura.
"""

from skimage.feature import local_binary_pattern
import numpy as np
import cv2

from src.utils.constants import NUMPOINTS, RADIUS


class LocalBinaryPattern:

    def __init__(self, numPoints: int = NUMPOINTS, radius: int = RADIUS):

        self.numPoints = numPoints
        self.radius = radius

    def extract_lbp_features(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:

        # de RGB a HSV
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask_bool = mask.astype(bool) if mask is not None else None

        features = []
        # se calcula sobre el canal de intensidad (V) en escala de grises
        lbp = local_binary_pattern(img_hsv[:, :, 2], P=self.numPoints, R=self.radius, method='uniform')
        # Sí se propone la máscara se usa la región de interés
        if mask_bool is None:
            lbp_values = lbp.ravel()
        else:
            lbp_values = lbp[mask_bool]

        # density=True --> el histograma queda normalizado
        hist, _ = np.histogram(lbp_values,
                               bins=self.numPoints + 2,
                               range=(0, self.numPoints + 2), density=True)

        features.append(hist)

        return np.concatenate(features)