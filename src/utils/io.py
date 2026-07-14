"""
Utilidades de entrada/salida: lectura de anotaciones COCO, conversión de
bounding boxes, localización y carga de imágenes/máscaras (robusta frente a
rutas con caracteres no ASCII) y volcado de resultados a disco.
"""

import json
import os
import numpy as np
import cv2
from pathlib import Path

def load_coco_annotations(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Agrupar las bounding boxes por image_id (una imagen puede tener varias).
    ann_by_image = {}
    for ann in data['annotations']:
        img_id = ann["image_id"]
        ann_by_image.setdefault(img_id, []).append(ann["bbox"])

    return data["images"], ann_by_image

def coco_bbox_to_xyxy(bbox):
    # COCO usa [x, y, w, h]; se convierte a [x_min, y_min, x_max, y_max] para MedSAM.
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h], dtype=np.float32)

def find_image_path(root_dir: str, filename: str) -> str | None:
    # Búsqueda recursiva del fichero por nombre.
    for root, _, files in os.walk(root_dir):
        if filename in files:
            return os.path.join(root, filename)
    return None


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Imagen en formato RGB y mask en grayscale"""
    return cv2.bitwise_and(image,image, mask= mask)

def save_json(obj, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

def prepare_output_dir(config: dict) -> Path:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_mask(path: str):
    # np.fromfile + imdecode en lugar de cv2.imread: soporta rutas con
    # caracteres Unicode, que cv2.imread no maneja bien en algunos sistemas.
    mask = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return mask

def load_image(path: str) -> np.ndarray:
        # Mismo patrón fromfile+imdecode que load_mask (rutas Unicode).
        img_bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
        # OpenCV lee en BGR; el resto del pipeline trabaja en RGB.
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)