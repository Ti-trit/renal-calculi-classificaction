"""
Extracción de embeddings CNN y proyección UMAP.

``extract_cnn_embeddings`` recupera la representación de 256-d que alimenta a la
última capa lineal de la red (mediante un forward hook), soportando tanto la
ResNet50 pura como el modelo híbrido. ``compute_umap`` estandariza esas
características y las proyecta a baja dimensión con UMAP para su visualización.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import umap
from src.utils.constants import SEED


@torch.no_grad()
def extract_cnn_embeddings(model, loader, device, num_classes=5):
    """
    Extrae la representación de 256-d (entrada a la capa lineal final)
    usando un forward hook.

    """
    model.eval()
    model.to(device)

    # Localizar la capa lineal final
    # Se identifica por tener out_features == num_classes (la capa de salida).
    final_linear = None
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.out_features == num_classes:
            final_linear = module
    if final_linear is None:
        raise RuntimeError("No se encontró la capa lineal final.")

    # El hook captura la ENTRADA de la capa final (el vector 256-d), que es el
    # embedding buscado, no su salida (los logits).
    captured = {}

    def hook(_module, inp, _out):
        captured["emb"] = inp[0].detach().cpu()

    handle = final_linear.register_forward_hook(hook)

    embeddings, labels_all = [], []

    for batch in loader:
        # El nº de elementos del batch distingue ResNet50 pura de híbrido.
        if len(batch) == 2:
            # ResNet-50
            images, labels = batch
            handcrafted = None
        else:
            # modelo híbrido
            images, handcrafted, labels = batch
            handcrafted = handcrafted.to(device).float()

        images = images.to(device)

        # Forward solo para disparar el hook; la salida del modelo se descarta.
        if handcrafted is not None:
            _ = model(images, handcrafted)
        else:
            _ = model(images)

        embeddings.append(captured["emb"].numpy())
        labels_all.append(
            labels.cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)
        )

    # Retirar el hook para no dejar efectos colaterales sobre el modelo.
    handle.remove()
    return np.vstack(embeddings), np.concatenate(labels_all)


def compute_umap(features, n_components=2, n_neighbors=15, min_dist=0.1):
    """Estandariza y proyecta con UMAP a n_components."""
    # Estandarizar antes de UMAP evita que las escalas dominen las distancias.
    X = StandardScaler().fit_transform(features)
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=SEED,
    )
    return reducer.fit_transform(X)