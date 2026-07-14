"""
Clasificador basado en ResNet50 con transfer learning.

Usa una ResNet50 preentrenada en ImageNet con el backbone congelado y solo una
cabeza de clasificación entrenable. El ``forward`` acepta un argumento
``handcrafted`` que ignora, para compartir interfaz con el modelo híbrido y
poder usar el mismo bucle de entrenamiento/inferencia con ambos.
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

from src.utils.constants import DROPOUT


class ResNet50Classifier(nn.Module):
    def __init__(
            self,
            num_classes: int,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.model = self.build_model()

    def build_model(self) -> nn.Module:
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        model = models.resnet50(weights=weights)

        # congelar backbone
        # Transfer learning: se entrena solamente la cabeza (fc)
        for param in model.parameters():
            param.requires_grad = False

        # Sustituir la capa fc original por una cabeza propia para num_classes
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT),
            nn.Linear(256, self.num_classes),
        )
        return model

    # se añade handcrafted para compatibilidad
    def forward(self, x: torch.Tensor, handcrafted: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.model(x)

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)