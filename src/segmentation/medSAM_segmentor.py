"""
Segmentación de imágenes con MedSAM (variante médica de SAM).

Envuelve el modelo SAM ViT-B para segmentar cálculos a partir de las bounding
boxes de las anotaciones COCO: preprocesa la imagen a 1024x1024, obtiene el
embedding del image encoder y, usando cada bbox como prompt, decodifica la
máscara. Las máscaras de todas las bboxes de una imagen se combinan (OR) en una
única máscara binaria.
"""

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry

from src.utils.io import coco_bbox_to_xyxy

class MedSAMSegmentor:


    def __init__(self, checkpoint: str, device: str= None) :
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model= self.load_model(checkpoint)
        self.image_size= 1024  # SAM opera sobre imágenes de 1024x1024.

    def load_model(self, checkpoint:str):
        # MedSAM se construye sobre la arquitectura ViT-B de SAM.
        model = sam_model_registry["vit_b"](checkpoint=checkpoint)
        model.to(self.device)
        model.eval()
        return model

    def preprocess(self, image_rgb: np.ndarray) -> torch.Tensor:
        # Redimensionar a 1024, normalizar min-max a [0, 1] (el +1e-8 evita
        # división por cero) y pasar a tensor CHW con dimensión de batch.
        image_1024 = cv2.resize(image_rgb, (self.image_size, self.image_size))
        image_1024 = (image_1024 - image_1024.min()) / (image_1024.max() - image_1024.min() + 1e-8)
        return torch.tensor(image_1024).permute(2, 0, 1).float().unsqueeze(0).to(self.device)

    def get_image_embedding(self, image_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.image_encoder(image_tensor)

    @torch.no_grad()
    def inference(self, img_embed, box_1024, height, width):
        box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=self.device)
        # Asegurar la forma (B, 1, 4) que espera el prompt_encoder para las cajas.
        if len(box_torch.shape) == 2:
            box_torch = box_torch[:, None, :]

        # La caja actúa como prompt.
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=None, boxes=box_torch, masks=None
        )
        low_res_logits, _ = self.model.mask_decoder(
            image_embeddings=img_embed,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        # Logits -> probabilidad; se reescala al tamaño original y se umbraliza
        # a 0.5 para obtener la máscara binaria.
        low_res_pred = torch.sigmoid(low_res_logits)
        low_res_pred = low_res_pred.squeeze().cpu().numpy()
        medsam_seg = (cv2.resize(low_res_pred, (width, height)) > 0.5).astype(np.uint8)
        return medsam_seg


    def segment_image(self, image_rgb, anns) :
        H, W= image_rgb.shape[:2]
        image_tensor = self.preprocess(image_rgb)
        # El embedding de la imagen solo depende de la imagen (no de las cajas),
        # así que se calcula una única vez y se reutiliza para todas las bboxes.
        img_embed = self.get_image_embedding(image_tensor)
        mask_total = np.zeros((H, W), dtype=bool)
        for ann_bbox in anns:
            bbox_xyxy = coco_bbox_to_xyxy(ann_bbox)
            # Escalar la caja de las coordenadas originales al espacio 1024x1024
            bbox_1024 = bbox_xyxy * np.array([
                self.image_size / W, self.image_size / H, self.image_size / W, self.image_size / H
            ], dtype=np.float32)

            mask = self.inference(
                img_embed, bbox_1024[None], H, W
            ).astype(bool)

            # Unir las máscaras de todas las cajas de la imagen
            mask_total |= mask

        return mask_total