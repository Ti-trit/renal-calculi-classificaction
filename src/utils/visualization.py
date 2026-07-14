"""
Funciones de visualización del proyecto.

Reúne las utilidades gráficas usadas a lo largo del pipeline: superposición de
máscara sobre la imagen, distribución de dimensiones del dataset, matriz de
confusión, comparativa de métricas por configuración (p. ej. barrido de LBP) y
proyecciones UMAP 2D de embeddings, tanto individuales como comparando varios
modelos lado a lado. La mayoría de funciones devuelven la figura de Matplotlib
para poder guardarla o insertarla en la memoria.
"""
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay
from src.analysis.embeddings import compute_umap
from pathlib import Path

from matplotlib.lines import Line2D



def get_overlay_mask(image_rgb, mask) :
    overlay = image_rgb.copy()
    overlay[mask] = [255, 0, 0]
    return cv2.addWeighted(image_rgb, 0.6, overlay, 0.4, 0)

def plot_dimensions(dimensions: dict):
    labels = [f"{h}x{w}" for h, w, c in dimensions.keys()]
    counts = list(dimensions.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, counts, color=sns.color_palette("Blues_d", len(labels)))

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom', fontsize=11)

    ax.set_title("Distribución de dimensiones de imagen")
    ax.set_xlabel("DimensiÓn (HxW)")
    ax.set_ylabel("Total")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    return fig
    
def plot_confusion_matrix(y_true, y_pred, class_names, output_dir: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names, cmap="Blues", ax=ax,
    )
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig((output_dir / title).with_suffix(".png"), dpi=150)
    plt.show()

def plot_lbp_multi_metric(df, labels:list, xlabel: str, ylabel:str, title:str,metrics: list = ("f1_weighted", "accuracy", "auc_weighted")):
    
    n_metrics = len(metrics)
    width = 0.8 / n_metrics
    
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(df))
    
    for i, metric in enumerate(metrics):
        means = df[f"{metric}_mean"].values * 100
        stds = df[f"{metric}_std"].values * 100
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(
            x + offset, means, width, yerr=stds, capsize=3,
            label=metric.replace("_", " ").title(), alpha=0.85,
        )
        
        for bar, mean, std in zip(bars, means, stds):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                mean + std + 0.5,
                f"{mean:.0f}",
                ha="center", va="bottom",
                fontsize=8, rotation=0,
            )
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(70, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    return fig








def plot_umap_2d(embedding, labels, class_names, cmap, title=None, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    labels = np.asarray(labels)
    for cls_idx, cls_name in enumerate(class_names):
        m = labels == cls_idx
        if not m.any():
            continue
        ax.scatter(
            embedding[m, 0], embedding[m, 1],
            color=cmap(cls_idx), alpha=0.6, s=25, label=cls_name,
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    if title:
        ax.set_title(title)
    return ax

def plot_models_umap_2d(features_dict, class_names, suptitle, save_path=None):
    n = len(features_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), squeeze=False)
    axes = axes[0]

    cmap = plt.get_cmap("tab10" if len(class_names) <= 10 else "tab20")

    for ax, (name, (feats, labels)) in zip(axes, features_dict.items()):
        emb = compute_umap(feats, n_components=2)
        plot_umap_2d(
            emb, labels, class_names, cmap=cmap,
            title=f"{name} ({feats.shape[1]}-d)", ax=ax,
        )

    handles = [
        Line2D([], [], marker="o", linestyle="", color=cmap(i), label=c, alpha=0.8)
        for i, c in enumerate(class_names)
    ]
    axes[0].legend(handles=handles, title="Clase", loc="upper right", fontsize=8)

    fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig