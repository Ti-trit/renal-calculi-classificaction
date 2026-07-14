"""
Inferencia del clasificador basado en ResNet50.

Este módulo expone una única función, predict, que recorre un DataLoader en
modo evaluación y devuelve las predicciones (y, opcionalmente, las
probabilidades por clase) para todo el conjunto. Se usa tras el entrenamiento,
una vez recargado el mejor checkpoint, para evaluar sobre el conjunto de test de
cada fold.

Al igual que el bucle de entrenamiento, soporta con el mismo código la ResNet50
pura y la variante híbrida (imagen + características handcrafted) gracias al
desempaquetado flexible de cada batch (images, *extra, labels).
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.resnet50 import ResNet50Classifier


@torch.no_grad()
def predict(
    model: ResNet50Classifier,
    loader: DataLoader,
    device: str,
    return_probs: bool = False,
) -> tuple:

    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        images, *extra, labels= batch
        handcrafted = extra[0].to(device) if extra else None
        images = images.to(device, non_blocking=True)
        logits = model(images,handcrafted )
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(
            labels.numpy() if hasattr(labels, 'numpy') else np.array(labels)
        )
        if return_probs:
            all_probs.append(probs.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    if return_probs:
        return preds, labels, np.concatenate(all_probs)
    return preds, labels

