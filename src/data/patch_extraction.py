"""
Extracción de parches centrados en la región del cálculo.

``extract_patches_centered`` muestrea parches cuyo centro cae dentro de la
máscara (y que no se salen de la imagen), rechazando los que quedan demasiado
cerca de otro centro ya elegido o que no alcanzan una cobertura mínima de
cálculo. ``build_patch_dataset`` aplica esa extracción a todo un DataFrame,
guarda parche y máscara en disco replicando la estructura por clase y devuelve un
DataFrame de metadatos (una fila por parche) que conserva la imagen de origen.
"""

from typing import Optional
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from src.utils.constants import *
from src.utils.io import load_image, load_mask


def extract_patches_centered(
        img_path: str,
        mask_path: str,
        patch_size: int,
        num_patches: int,
        min_coverage: float,
        min_distance_between_centers: int,
        max_attempts: int,
) -> list[dict]:
    """
    Extrae parches centrados en un píxel dada dentro de la región que ocupa el cálculo.
    Se devuelve el parche y su máscara correspondiente.

    """
    # Cargar la imágen y la máscara
    img = load_image(img_path)
    mask = load_mask(mask_path)

    # Binarizar máscara
    mask_bin = (mask > 0).astype(np.uint8)

    H, W = mask_bin.shape
    half = patch_size // 2

    # Buscar los píxels candidatos del centro de un parche
    # Se exluyen los puntos que pueden generar parches que salen de la imagen
    valid_region = np.zeros_like(mask_bin)
    valid_region[half:H - half, half:W - half] = mask_bin[half:H - half, half:W - half]
    candidate_ys, candidate_xs = np.where(valid_region > 0)

    if len(candidate_ys) == 0:
        # La piedra es demasiado pequeña o está demasiado cerca del borde para este tamaño de parche.
        return []

    # Semilla fija
    rng = np.random.default_rng(SEED)
    selected_centers = []
    patches = []
    attempts = 0

    # Repetir hasta reunir num_patches o agotar max_attempts
    while len(patches) < num_patches and attempts < max_attempts:
        attempts += 1

        # Centro aleatorio
        idx = rng.integers(0, len(candidate_ys))
        cy, cx = candidate_ys[idx], candidate_xs[idx]

        # Rechazar centros demasiado próximos a otro ya aceptado (menos solape)
        if selected_centers:
            centers_arr = np.array(selected_centers)
            dists = np.sqrt(np.sum((centers_arr - [cy, cx]) ** 2, axis=1))
            if np.min(dists) < min_distance_between_centers:
                continue

        # Extracción del parche y su máscara
        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half
        patch_img = img[y0:y1, x0:x1]
        patch_mask = mask_bin[y0:y1, x0:x1]

        # Cumple la cobertura mínima?
        # coverage = fracción del parche ocupada por el cálculo.
        coverage = float(patch_mask.mean())
        if coverage < min_coverage:
            continue

        patches.append({
            "patch": patch_img.astype(np.uint8),
            "patch_mask": patch_mask.astype(np.uint8),
            "coverage": coverage
        })
        selected_centers.append((cy, cx))

    return patches


def build_patch_dataset(
        df: pd.DataFrame,
        output_dir: Path,
        patch_size: int,
        num_patches: int,
        min_coverage: float,
        min_distance_between_centers: int,
        max_attempts: int,
        label_cols: Optional[list[str]] = None,
        class_col: str = "folder",
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Si no se especifican, se arrastran todas las columnas salvo rutas e id.
    if label_cols is None:
        excluded = {IMG_COL, MASK_COL, STONE_ID_COL}
        label_cols = [c for c in df.columns if c not in excluded]

    records = []
    skipped = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting patches"):
        img_path = Path(row[IMG_COL])
        mask_path = Path(row[MASK_COL])
        stone_id = str(row[STONE_ID_COL])
        cls = str(row[class_col])

        # Carpetas específicas de esta clase
        class_dir = output_dir / cls
        patches_dir = class_dir / "patches"
        masks_dir = class_dir / "masks"
        patches_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        patches = extract_patches_centered(
            img_path=img_path,
            mask_path=mask_path,
            patch_size=patch_size,
            num_patches=num_patches,
            min_coverage=min_coverage,
            min_distance_between_centers=min_distance_between_centers,
            max_attempts=max_attempts,
        )

        # Imagen sin parches válidos (demasiado pequeña) -> se registra y salta
        if not patches:
            skipped.append(stone_id)
            continue

        for i, p in enumerate(patches):
            # se construye el id del parche a partir del stem de la imagen de la que deriva
            # esto permitirá agregar los resultados de los parches a nivel de imagen
            patch_id = f"{img_path.stem}_p{i:02d}"
            patch_path = patches_dir / f"{patch_id}.png"
            patch_mask_path = masks_dir / f"{patch_id}_mask.png"

            # La máscara se guarda escalada a 0/255 para poder visualizarla
            Image.fromarray(p["patch"]).save(patch_path)
            Image.fromarray(
                (p["patch_mask"] * 255).astype(np.uint8)
            ).save(patch_mask_path)

            record = {
                "patch_id": patch_id,
                "patch_path": str(patch_path),
                "patch_mask_path": str(patch_mask_path),
                "stone_id": stone_id,
                "source_image": str(img_path),
                "source_mask": str(mask_path),
            }
            # Propagar las columnas de etiqueta/metadatos de la imagen al parche
            for col in label_cols:
                record[col] = row[col]
            records.append(record)

    patches_df = pd.DataFrame(records)
    print(f"\n[Patch extraction summary]")
    print(f"  Source images:   {len(df)}")
    print(f"  Patches created: {len(patches_df)}")
    print(f"  Skipped images:  {len(skipped)} too small")
    if skipped:
        print(
            f"  Skipped IDs:     {skipped[:5]}"
            f"{'...' if len(skipped) > 5 else ''}"
        )
    return patches_df