"""
Preprocesamiento del dataset a nivel de ficheros.

Agrupa las utilidades que preparan el dataset en disco antes del pipeline:
conversión de TIFF a PNG, normalización de nombres de archivo (sin acentos ni
espacios), inventario de dimensiones de las imágenes, extracción del ``stone_id``
desde el nombre, y redimensionado uniforme de imágenes y máscaras (rotando las
que vienen con las dimensiones invertidas) replicando la estructura de carpetas.
"""

from pathlib import Path
import pandas as pd
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
import unicodedata
import re


def from_tif_to_png(images_root: str):
    for root, dirs, files in os.walk(images_root):
        for file in files:
            if file.lower().endswith(".tif"):
                tif_path = os.path.join(root, file)
                try:
                    # Convertir a RGB y guardar como PNG.
                    image_tif = Image.open(tif_path)
                    png_image = image_tif.convert('RGB')
                    png_path = os.path.join(root, os.path.splitext(file)[0]) + ".png"
                    png_image.save(png_path, format='PNG')
                    if os.path.exists(png_path):
                        os.remove(tif_path)
                except Exception as e:
                    print(f"Error with image {tif_path}: {e}")


def normalize_name(root_path: str):
    for root, dirs, files in os.walk(root_path):
        for file in files:
            old_path = os.path.join(root, file)
            name, ext = os.path.splitext(file)

            # NFD + encode/ignore elimina los acentos (á -> a) descomponiendo
            # el carácter y descartando la marca diacrítica.
            name_normalized = unicodedata.normalize('NFD', name)
            name_normalized = name_normalized.encode('ascii', 'ignore').decode('utf-8')
            name_normalized = name_normalized.replace(" ", "_")
            name_normalized = name_normalized.replace("&", "and")
            new_path = os.path.join(root, name_normalized + ext)
            os.rename(old_path, new_path)


def get_images_dimensions(root_path: str, extensions=('.jpg', '.png')) -> pd.DataFrame:
    # Cuenta cuántas imágenes hay de cada forma (shape) en todo el dataset.
    dimensions = defaultdict(int)
    root = Path(root_path)
    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue

        for view_dir in class_dir.iterdir():
            if not view_dir.is_dir():
                continue

            for img_file in view_dir.iterdir():
                if img_file.is_dir():  # carpeta MASKS
                    continue
                if img_file.suffix.lower() not in extensions:
                    continue

                im = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), cv2.IMREAD_COLOR)
                if im is None:
                    print(f"[SKIP] No se pudo leer: {img_file}")
                    continue
                # ancho x altura x canal
                dimensions[im.shape] += 1
    return dimensions


def extract_stone_id(img_file: Path) -> str | None:
    """
    Extrae el stone_id del nombre del archivo.

    Formato : <clase>_<identificador>[_<info>][_<instancia>].<ext>
    <info> y <instancia> pueden estar en orden girado.
    El stone_id es <clase>_<identificador>.
    """
    # Captura <clase>_<número><letras opcionales> e ignora el resto del nombre.
    match = re.match(r"^(.+?_\d+[A-Za-z]*)(?:_.*)?$", img_file.stem)
    return match.group(1) if match else None


def resize_to_target(image: np.ndarray, mask: np.ndarray, size=tuple) -> tuple[np.ndarray, np.ndarray]:
    target_h, target_w = size  # (1372, 1792)
    h, w = image.shape[:2]

    if h == target_h and w == target_w:
        # Tamaño correcto, no hacer nada
        pass

    elif h == target_w and w == target_h:
        # Dimensiones invertidas → rotar 90°
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)

    else:
        # Tamaño distinto --> escalar
        # INTER_AREA para la imagen; INTER_NEAREST en la máscara para no crear
        # etiquetas intermedias entre fondo y cálculo.
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    return image, mask


def resize_images(root_path: str, output_path: str = "DATASET/RESIZED", extensions=('.jpg', '.png')):
    root = Path(root_path)
    output_root = Path(output_path)

    n_processed = 0
    n_skipped = 0

    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue
        for view_dir in class_dir.iterdir():
            if not view_dir.is_dir():
                continue
            # Construir paths espejando la estructura
            target_img_dir = output_root / class_dir.name / view_dir.name
            target_mask_dir = target_img_dir / "MASKS"
            target_img_dir.mkdir(parents=True, exist_ok=True)
            target_mask_dir.mkdir(parents=True, exist_ok=True)
            for img_file in view_dir.iterdir():
                if img_file.is_dir():
                    continue
                if img_file.suffix.lower() not in extensions:
                    continue

                # Se salta la imagen si no tiene máscara emparejada.
                mask_path = view_dir / "MASKS" / (img_file.stem + "_mask.png")
                if not mask_path.exists():
                    print(f"AVISO: máscara no encontrada para {img_file.name}")
                    n_skipped += 1
                    continue


                img = cv2.imdecode(np.fromfile(img_file, dtype=np.uint8), cv2.IMREAD_COLOR)
                mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

                if img is None or mask is None:
                    print(f"AVISO: fallo al leer {img_file.name}")
                    n_skipped += 1
                    continue

                # Redimensionado
                resized_image, resized_mask = resize_to_target(img, mask)

                target_img_path = target_img_dir / img_file.name
                target_mask_path = target_mask_dir / mask_path.name

                # imencode + tofile: contrapartida de imdecode/fromfile para
                # escribir respetando rutas con caracteres Unicode.
                _, img_buf = cv2.imencode(img_file.suffix, resized_image)
                img_buf.tofile(target_img_path)

                _, mask_buf = cv2.imencode(mask_path.suffix, resized_mask)
                mask_buf.tofile(target_mask_path)

                n_processed += 1

    print(f"\nProcesadas: {n_processed} | Saltadas: {n_skipped}")