"""
Balanceo del conjunto de parches por sobremuestreo.

``balance_patches_by_oversampling`` lleva cada clase minoritaria hasta un objetivo
de parches por clase generando parches adicionales a partir de sus imágenes de
origen. ``generate_additional_patches`` hace el trabajo por clase: reparte los
parches que faltan entre las imágenes disponibles en pasadas sucesivas, los
guarda replicando la estructura de carpetas del dataset y devuelve sus metadatos.
Los parches nuevos se marcan con ``is_augmented=True``.
"""

import pandas as pd
from pathlib import Path
import cv2
from src.utils.constants import LABEL_COL
from src.data.patch_extraction import extract_patches_centered


def balance_patches_by_oversampling(
        patches_df: pd.DataFrame,
        target_n_per_class: int,
        patch_size: int,
        min_coverage: float,
        output_dir: Path,
) -> pd.DataFrame:
    """Balancea un conjunto de datos de parches aplicando
    sobremuestreo a las clases minoritarias."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = patches_df[LABEL_COL].value_counts()
    print(f"Conteos actuales por clase:")
    print(counts.to_string())

    # Sin objetivo explícito, se iguala a la clase mayoritaria.
    if target_n_per_class is None:
        target_n_per_class = int(counts.max())
    print(f"Target: {target_n_per_class} patches por clase\n")

    # Los parches originales se conservan marcados como no augmentados.
    original_df = patches_df.copy()
    original_df["is_augmented"] = False

    additional_rows = []

    for cls in counts.index:
        n_current = counts[cls]
        n_needed = target_n_per_class - n_current

        # La clase ya cumple el objetivo: nada que generar.
        if n_needed <= 0:
            print(f"Clase {cls}: ya tiene {n_current} patches, saltando.")
            continue

        print(f"Clase {cls}: {n_current} → {target_n_per_class} ({n_needed} adicionales)")

        # Imágenes únicas de la clase (una fila por source_image), con sus metadatos.
        cls_patches = patches_df[patches_df[LABEL_COL] == cls]
        images_cls = (
            cls_patches[["source_image", "source_mask", "stone_id"]]
            .drop_duplicates(subset="source_image")
            .reset_index(drop=True)
        )

        # Recuperar folder/view si existen, para replicar la estructura al guardar.
        extra_cols = [c for c in ("folder", "view") if c in cls_patches.columns]
        if extra_cols:
            extras = (
                cls_patches[["source_image"] + extra_cols]
                .drop_duplicates(subset="source_image")
            )
            images_cls = images_cls.merge(extras, on="source_image", how="left")

        print(f"  Imágenes disponibles: {len(images_cls)}")

        # Reparto de los parches necesarios entre las imágenes (+1 como margen).
        patches_per_image = max(1, (n_needed // len(images_cls)) + 1)

        patches_generated = generate_additional_patches(
            images_cls=images_cls,
            patches_per_image=patches_per_image,
            target_total=n_needed,
            patch_size=patch_size,
            min_coverage=min_coverage,
            output_dir=output_dir,
            class_label=cls,
            extra_cols=extra_cols,
        )

        additional_rows.extend(patches_generated)

    # Unir originales + augmentados (si se han generado).
    if additional_rows:
        additional_df = pd.DataFrame(additional_rows)
        additional_df["is_augmented"] = True
        balanced_df = pd.concat([original_df, additional_df], ignore_index=True)
    else:
        balanced_df = original_df

    print(balanced_df[LABEL_COL].value_counts().to_string())
    print(f"\nTotal: {len(balanced_df)} patches "
          f"({balanced_df['is_augmented'].sum()} augmentados)")

    return balanced_df


def generate_additional_patches(
        images_cls: pd.DataFrame,
        patches_per_image: int,
        target_total: int,
        patch_size: int,
        min_coverage: float,
        output_dir: Path,
        class_label,
        extra_cols: list,
) -> list[dict]:
    """
    Balancea las clases extrayendo patches adicionales en clases minoritarias rebajando la cobertura mínima a min_coverage.
    Los patches generados se guardan replicando la estructura de carpetas
    del dataset original:
        output_dir/{class_label}/{view}/patches/{patch_id}.png
        output_dir/{class_label}/{view}/masks/{patch_id}_mask.png

    """
    patches_metadata = []
    total_generated = 0
    pass_idx = 0

    # Pasadas sucesivas sobre las imágenes hasta reunir target_total parches
    while total_generated < target_total:
        pass_idx += 1
        progress_in_pass = 0

        for _, img_row in images_cls.iterrows():
            if total_generated >= target_total:
                break

            img_path = Path(img_row["source_image"])
            mask_path = Path(img_row["source_mask"])

            view = img_row.get("view") if "view" in extra_cols else None
            if view is not None:
                class_view_dir = output_dir / str(class_label) / str(view)
            else:
                class_view_dir = output_dir / str(class_label)

            patches_dir = class_view_dir / "patches"
            masks_dir = class_view_dir / "masks"
            patches_dir.mkdir(parents=True, exist_ok=True)
            masks_dir.mkdir(parents=True, exist_ok=True)

            n_for_this = min(
                patches_per_image,
                target_total - total_generated,
            )

            # min_distance_between_centers=0 permite solape entre parches, para
            # poder generar más muestras de una misma imagen minoritaria.
            try:
                patches = extract_patches_centered(
                    img_path=img_path,
                    mask_path=mask_path,
                    patch_size=patch_size,
                    num_patches=n_for_this,
                    min_coverage=min_coverage,
                    min_distance_between_centers=0,
                    max_attempts=n_for_this * 20,
                )
            except FileNotFoundError as e:
                print(f"    Error: {e}, saltando imagen")
                continue

            if not patches:
                continue

            for patch_data in patches:
                # patch_id incluye el stem de la imagen para evitar colisiones
                # entre views distintas con mismo stone_id
                patch_id = (
                    f"{img_path.stem}_aug_p{pass_idx}_{total_generated:05d}"
                )
                patch_path = patches_dir / f"{patch_id}.png"
                mask_out_path = masks_dir / f"{patch_id}_mask.png"

                cv2.imwrite(
                    str(patch_path),
                    cv2.cvtColor(patch_data["patch"], cv2.COLOR_RGB2BGR),
                )
                cv2.imwrite(
                    str(mask_out_path),
                    patch_data["patch_mask"] * 255,
                )

                row = {
                    "patch_id": patch_id,
                    "patch_path": str(patch_path),
                    "patch_mask_path": str(mask_out_path),
                    "stone_id": img_row["stone_id"],
                    "source_image": str(img_row["source_image"]),
                    "source_mask": str(img_row["source_mask"]),
                    LABEL_COL: class_label,
                }
                for col in extra_cols:
                    row[col] = img_row.get(col, None)

                patches_metadata.append(row)
                total_generated += 1
                progress_in_pass += 1

        # Si una pasada completa no añade nada, no hay margen para más: cortar.
        if progress_in_pass == 0:
            print(f"    AVISO: no se pueden generar más patches para clase "
                  f"{class_label}. Generados: {total_generated}/{target_total}")
            break

    return patches_metadata