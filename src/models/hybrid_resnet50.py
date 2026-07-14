"""
Modelo híbrido ResNet50 + descriptores handcrafted.

Combina las características visuales de una ResNet50 preentrenada (backbone
congelado, sin la capa fc) con un vector handcrafted (LBP+HSV) concatenándolos
antes de la cabeza de clasificación. El backbone y el clasificador se agrupan en
un ``nn.ModuleDict`` bajo ``self.model``, misma convención que
``ResNet50Classifier``, para que el guardado/carga de checkpoints
(``model.model.state_dict()``) funcione igual en ambos modelos.
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models

from src.utils.constants import DROPOUT


class HybridResNet50(nn.Module):

    def __init__(self, num_classes: int, n_handcrafted: int):
        super().__init__()
        self.num_classes = num_classes
        self.n_handcrafted = n_handcrafted
        self.backbone, self.classifier = self.build_model()
        # Agrupar bajo self.model
        self.model= nn.ModuleDict({
            'backbone': self.backbone,
            'classifier': self.classifier
        })

    def build_model(self) -> tuple[nn.Module, nn.Module]:
        """
               Cabecera adaptada:
               [2048 (ResNet backbone) || n_handcrafted] → 256 → BN → ReLU → Dropout → num_classes
           """
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        resnet = models.resnet50(weights=weights)
        # Backbone congelado
        for param in resnet.parameters():
            param.requires_grad = False
        backbone = nn.Sequential(*list(resnet.children())[:-1])  # → (B, 2048, 1, 1)
        in_features = resnet.fc.in_features  # 2048
        # La entrada del clasificador suma las features visuales y las handcrafted.
        classifier = nn.Sequential(
            nn.Linear(in_features + self.n_handcrafted, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT),
            nn.Linear(256, self.num_classes),
        )
        return backbone, classifier

    def forward(self, x: torch.Tensor, handcrafted: Optional[torch.Tensor] = None) -> torch.Tensor:

        # Backbone -> aplanar a (B, 2048), concatenar el vector handcrafted y
        # pasar el vector combinado por la cabeza de clasificación.
        feat= self.model['backbone'](x).flatten(1)
        combined= torch.cat([feat, handcrafted], dim= 1)
        return self.model['classifier'](combined)

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)