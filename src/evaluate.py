"""
Cálculo, agregación y reporte de métricas de clasificación.

Este módulo reúne toda la lógica de evaluación del proyecto: métricas por
predicción (accuracy, precision/recall/F1 weighted, AUC, métricas por clase,
matriz de confusión y classification_report), su persistencia a disco, la
agregación entre folds de la validación cruzada y, para el caso de patches, la
agregación de predicciones al nivel de imagen mediante soft voting.

Convenciones:
    - Las métricas promediadas usan ``average="weighted"`` (ponderado por
      soporte), coherente con el resto del pipeline.
    - El AUC se adapta a binario (probabilidad de la clase positiva) o
      multiclase (one-vs-rest, promedio weighted).
    - Las funciones ``*_per_image*`` asumen un DataFrame OOF a nivel de patch con
      columnas de probabilidad por clase (prefijo ``prob_``).
"""

import json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from src.utils.constants import LABEL_COL

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


def compute_auc(y_true: np.ndarray, y_proba: np.ndarray, num_classes: int) -> Optional[float]:
    """
    AUC con tratamiento adecuado según binario/multiclase.

    - Binario: usa las probabilidades de la clase positiva (índice 1).
    - Multiclase: one-vs-rest con promedio weighted.

    Es robusta frente a dos situaciones que harían fallar a ``roc_auc_score``:
    que alguna clase no aparezca en ``y_true`` (restringe las probabilidades a
    las clases presentes) y cualquier otro ``ValueError`` (devuelve ``None`` y
    avisa por consola en lugar de interrumpir la evaluación).

    Args:
        y_true: Etiquetas verdaderas (array 1-D).
        y_proba: Matriz de probabilidades ``(n_muestras, n_clases)``.
        num_classes: Número total de clases del problema.

    Returns:
        Optional[float]: AUC como float, o ``None`` si no se pudo calcular.
    """
    try:
        if num_classes == 2:
            return float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            # Manejar el caso donde alguna clase está ausente en y_true
            present = sorted(np.unique(y_true))
            if len(present) == num_classes:
                proba_used = y_proba
            else:
                # Restringir a las clases presentes
                proba_used = y_proba[:, present]
                print(f"  AVISO: AUC calculado sobre {len(present)}/{num_classes} clases presentes")
            return float(roc_auc_score(
                y_true, proba_used, multi_class="ovr", average="weighted",
            ))
    except ValueError as e:
        # Cualquier condición no soportada (p. ej. una sola clase presente) se
        # degrada a None sin cortar el flujo de evaluación.
        print(f"  AVISO: no se pudo calcular AUC: {e}")
        return None


def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        class_names: Optional[list] = None
) -> dict:
    """
    Calcula las métricas de clasificación con promedio ponderado

    Devuelve un diccionario con las métricas globales (accuracy y
    precision/recall/F1 weighted), el AUC si se pasan probabilidades, las
    métricas por clase, la matriz de confusión y el ``classification_report``
    en formato dict. Todo se serializa a tipos nativos para poder volcarse a
    JSON.

    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Etiquetas predichas.
        y_proba: Probabilidades por clase; si es ``None`` no se calcula AUC.
        class_names: Nombres de las clases; si es ``None`` se infiere el número
            de clases a partir de los valores presentes.

    Returns:
        dict: Diccionario de métricas listo para reportar o serializar.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    # Nº de clases: preferimos len(class_names); si no, lo inferimos de los datos.
    num_classes = len(class_names) if class_names is not None else len(np.unique(np.concatenate([y_true, y_pred])))

    # Métricas globales (ponderadas por soporte).
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    # adaptar AUC a multiclase
    # Solo se añade si hay probabilidades disponibles.
    if y_proba is not None:
        metrics["auc_weighted"] = compute_auc(y_true, y_proba, num_classes)

    # Métricas por cada clase
    # average=None devuelve un valor por clase; se convierte a lista para JSON.
    metrics["per_class"] = {
        "precision": precision_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "recall": recall_score(y_true, y_pred, average=None, zero_division=0).tolist(),
        "f1": f1_score(y_true, y_pred, average=None, zero_division=0).tolist(),
    }
    if class_names is not None:
        metrics["per_class"]["class_names"] = class_names

    # Matriz de confusión y classification_report
    # Ambos como estructuras serializables (lista y dict, respectivamente).
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    metrics["classification_report"] = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return metrics


def save_metrics(metrics: dict, output_dir: str, name: str = "metrics.json"):
    """Guarda un diccionario de métricas como JSON.

    Args:
        metrics: Diccionario de métricas (p. ej. la salida de ``compute_metrics``).
        output_dir: Directorio de destino.
        name: Nombre del fichero JSON.

    Note:
        ``default=str`` permite serializar tipos no nativos de JSON (por ejemplo
        objetos de NumPy que hayan quedado sin convertir).
    """
    path = Path(output_dir) / name
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)


def save_predictions(
        df: pd.DataFrame,
        preds: np.ndarray,
        output_dir: str,
        probs: Optional[np.ndarray] = None,
        name: str = "test_predictions.csv",
):
    """Guarda un CSV con las predicciones (y probabilidades) sobre un DataFrame.

    Copia ``df`` para no mutar el original, añade la columna ``pred`` y, si se
    pasan probabilidades, una columna ``prob_{c}`` por clase.

    Args:
        df: DataFrame base (metadatos de las muestras).
        preds: Predicciones a añadir como columna ``pred``.
        output_dir: Directorio de destino.
        probs: Probabilidades por clase, opcionales.
        name: Nombre del fichero CSV.
    """
    out = df.copy()
    out["pred"] = preds
    # Una columna de probabilidad por clase, indexada por su posición.
    if probs is not None:
        for c in range(probs.shape[1]):
            out[f"prob_{c}"] = probs[:, c]
    out.to_csv(Path(output_dir) / name, index=False)


def print_report(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        class_names: Optional[list] = None,
):
    """Imprime por consola un resumen legible de las métricas.

    Calcula las métricas con ``compute_metrics`` y muestra las globales, la
    matriz de confusión y el ``classification_report`` en formato texto.

    Args:
        y_true: Etiquetas verdaderas.
        y_pred: Etiquetas predichas.
        y_proba: Probabilidades por clase; habilita la línea de AUC si existen.
        class_names: Nombres de las clases para el reporte.
    """
    metrics = compute_metrics(y_true, y_pred, y_proba, class_names)

    print(f"Accuracy:              {metrics['accuracy']:.4f}")
    print(f"Recall weighted:        {metrics['recall_weighted']:.4f}")
    print(f"Precision weighted:   {metrics['precision_weighted']:.4f}")
    print(f"F1 weighted:           {metrics['f1_weighted']:.4f}")
    # El AUC solo se imprime si se calculó (no None).
    if "auc_weighted" in metrics and metrics["auc_weighted"] is not None:
        print(f"AUC weighted:          {metrics['auc_weighted']:.4f}")

    print(f"\nConfusion matrix:")
    print(np.array(metrics["confusion_matrix"]))

    print(f"\nClassification report:")
    print(classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0,
    ))


def aggregate_fold_metrics(fold_results: list[dict]) -> dict:
    """Agrega entre folds calculando media y desviación de cada métrica numérica.

    Recorre todas las claves presentes en los folds, ignora ``fold`` y las no
    numéricas (p. ej. ``per_class`` o ``confusion_matrix``), y produce
    ``{metrica}_mean`` y ``{metrica}_std`` para las que sí lo son.

    Args:
        fold_results: Lista de diccionarios de métricas, uno por fold.

    Returns:
        dict: Resumen con las medias y desviaciones estándar entre folds.
    """
    summary = {}
    # Reunir el conjunto de todas las métricas que aparecen en algún fold
    metric_names = set()
    for fr in fold_results:
        metric_names.update(fr.keys())

    for metric in metric_names:
        if metric == "fold":
            continue
        # Filtrar a valores escalares numéricos
        values = [fr.get(metric) for fr in fold_results]
        numeric = [v for v in values if isinstance(v, (int, float)) and v is not None]
        if numeric:
            summary[f"{metric}_mean"] = float(np.mean(numeric))
            summary[f"{metric}_std"] = float(np.std(numeric))
    return summary


def print_summary(summary):
    """Imprime el resumen de la validación cruzada como media ± std por métrica.

    Args:
        summary: Diccionario con claves ``{metrica}_mean`` / ``{metrica}_std``
            (la salida de ``aggregate_fold_metrics``).
    """
    print(f"\nRESULTADOS CV 5 -fold (media ± std entre folds):")
    metric_names = sorted({k.removesuffix("_mean") for k in summary if k.endswith("_mean")})
    for metric in metric_names:
        mean_v = summary[f"{metric}_mean"]
        std_v = summary[f"{metric}_std"]
        print(f"  {metric}: {mean_v:.4f} ± {std_v:.4f}")


def aggregate_per_image(
        patch_df: pd.DataFrame,
        patch_proba: np.ndarray,
        image_id_col: str = "source_image",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Agrega las predicciones de los patches al nivel de imagen mediante
    soft voting (promedio de probabilidades).

    Para cada imagen promedia (de forma uniforme) las probabilidades de sus
    patches y toma como predicción el argmax del promedio. Comprueba la
    consistencia de los datos: mismo número de filas en ``patch_df`` y
    ``patch_proba``, y una única etiqueta verdadera por imagen.

    Args:
        patch_df: DataFrame a nivel de patch (incluye ``LABEL_COL`` e
            ``image_id_col``).
        patch_proba: Probabilidades por patch, alineadas fila a fila con
            ``patch_df``.
        image_id_col: Columna que identifica la imagen de origen de cada patch.

    Returns:
        tuple: ``(image_ids, y_true_img, y_pred_img, y_proba_img)`` a nivel de
        imagen.

    Raises:
        ValueError: Si las longitudes no coinciden o una imagen tiene patches
            con etiquetas distintas.
    """
    if len(patch_df) != len(patch_proba):
        raise ValueError(
            f"patch_df y patch_proba deben tener el mismo número de filas: "
            f"{len(patch_df)} vs {len(patch_proba)}"
        )

    # Trabajamos sobre una copia con índice reseteado para indexar patch_proba
    # (así _patch_idx sirve para localizar cada fila dentro de patch_proba).
    df = patch_df.reset_index(drop=True).copy()
    df["_patch_idx"] = np.arange(len(df))

    image_ids = []
    y_true_img = []
    y_pred_img = []
    y_proba_img = []

    # sort=True garantiza un orden estable de las imágenes en la salida.
    for image_id, group in df.groupby(image_id_col, sort=True):
        indices = group["_patch_idx"].to_numpy()

        # Promedio uniforme de probabilidades sobre los patches de esta imagen
        proba_mean = patch_proba[indices].mean(axis=0)

        # Predicción agregada
        pred = int(np.argmax(proba_mean))

        # Etiqueta verdadera (debe ser la misma para todos los patches de una imagen)
        labels = group[LABEL_COL].unique()
        if len(labels) > 1:
            raise ValueError(
                f"La imagen {image_id} tiene patches con etiquetas distintas: {labels}"
            )

        image_ids.append(image_id)
        y_true_img.append(int(labels[0]))
        y_pred_img.append(pred)
        y_proba_img.append(proba_mean)

    return (
        np.array(image_ids),
        np.array(y_true_img),
        np.array(y_pred_img),
        np.array(y_proba_img),
    )


def compute_per_image_metrics(
        oof_df: pd.DataFrame,
        image_id_col: str = "source_image",
        class_names: Optional[list] = None,
        prob_prefix: str = "prob_",
) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    A partir del DataFrame OOF a nivel patch, agrega las predicciones
    al nivel de imagen y calcula las métricas correspondientes.

    Selecciona las columnas de probabilidad (por nombre de clase si se dan, o
    por prefijo ``prob_``), agrega a nivel de imagen con ``aggregate_per_image``
    y calcula las métricas con ``compute_metrics``.

    Args:
        oof_df: DataFrame OOF a nivel de patch con columnas de probabilidad.
        image_id_col: Columna identificadora de la imagen de origen.
        class_names: Nombres de clase; fijan el orden de las columnas de
            probabilidad si se proporcionan.
        prob_prefix: Prefijo de las columnas de probabilidad.

    Returns:
        tuple: ``(metrics, y_true_img, y_pred_img)`` a nivel de imagen.
    """
    # Orden de columnas de probabilidad: por class_names si existe; si no,
    # ordenación de todas las columnas con el prefijo.
    if class_names is not None:
        prob_cols = [f"{prob_prefix}{c}" for c in class_names]
    else:
        prob_cols = sorted(c for c in oof_df.columns if c.startswith(prob_prefix))

    patch_proba = oof_df[prob_cols].to_numpy()

    image_ids, y_true_img, y_pred_img, y_proba_img = aggregate_per_image(
        patch_df=oof_df,
        patch_proba=patch_proba,
        image_id_col=image_id_col,
    )

    metrics = compute_metrics(
        y_true=y_true_img,
        y_pred=y_pred_img,
        y_proba=y_proba_img,
        class_names=class_names,
    )

    return metrics, y_true_img, y_pred_img


def compute_per_image_metrics_by_fold(
        oof_df: pd.DataFrame,
        fold_col: str = "fold",
        image_id_col: str = "source_image",
        class_names: Optional[list] = None,
        prob_prefix: str = "prob_",
) -> tuple[list[dict], dict, np.ndarray, np.ndarray]:
    """
    Calcula métricas a nivel imagen para cada fold por separado y agrega.

    Agrupa el OOF por fold, calcula las métricas a nivel de imagen de cada uno
    con ``compute_per_image_metrics``, resume entre folds (media ± std) y
    concatena las etiquetas verdaderas y predichas de todos los folds.

    Args:
        oof_df: DataFrame OOF a nivel de patch con columna de fold.
        fold_col: Columna que identifica el fold.
        image_id_col: Columna identificadora de la imagen de origen.
        class_names: Nombres de las clases.
        prob_prefix: Prefijo de las columnas de probabilidad.

    Returns:
        tuple: ``(fold_metrics_list, summary, y_true_concat, y_pred_concat)``.
    """
    fold_metrics_list = []
    all_y_true = []
    all_y_pred = []

    # Un cálculo de métricas a nivel imagen por fold.
    for fold_id, fold_subset in oof_df.groupby(fold_col):
        m, y_true_fold, y_pred_fold = compute_per_image_metrics(
            oof_df=fold_subset,
            image_id_col=image_id_col,
            class_names=class_names,
            prob_prefix=prob_prefix,
        )
        m["fold"] = int(fold_id)
        fold_metrics_list.append(m)
        all_y_true.append(y_true_fold)
        all_y_pred.append(y_pred_fold)

    # Resumen entre folds + verdad/predicción concatenadas de todos ellos.
    summary = aggregate_fold_metrics(fold_metrics_list)
    return (
        fold_metrics_list,
        summary,
        np.concatenate(all_y_true),
        np.concatenate(all_y_pred),
    )