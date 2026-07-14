"""
Dataset de PyTorch y transformaciones para los cálculos renales.

``KidneyStoneDataset`` carga imagen y máscara, aplica la máscara para aislar el
cálculo y devuelve el tensor transformado junto con la etiqueta; si se le pasa un
diccionario ``handcrafted``, devuelve además el vector handcrafted emparejado por
clave (usado por el modelo híbrido). ``get_transform`` define las
transformaciones de train (con augmentation) y de evaluación, normalizando con
las estadísticas de ImageNet que espera la ResNet50 preentrenada.
"""
from typing import  Callable
from src.utils.constants import *
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from src.utils.io import load_mask, apply_mask, load_image
class KidneyStoneDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Callable,
        case:Case, 
        handcrafted:np.ndarray= None, 
        key_col: str= "match_key"
    ):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self.case= case
        self.handcrafted= handcrafted
        self.key_col= key_col

        

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_path= row[self.case.image_col]
        image = load_image(image_path)
    
        # Aplicar la máscara
        mask = load_mask(row[self.case.mask_col])
        if mask is None:
            raise FileNotFoundError(f"No se ha encontrado una màscara para la imagen : {image_path} ")
        else:
            image = apply_mask(image, mask)

        
        if self.transform:
            image = self.transform(image)  

        label = row[LABEL_COL]
        # Modo híbrido: adjuntar el vector handcrafted buscándolo por su clave.
        if self.handcrafted is not None:
            key= row[self.key_col]
            feats= torch.tensor(self.handcrafted[key], dtype= torch.float32)
            return image, feats, label
        return image, label
    


   



def get_transform(image_size: int , augment: bool = False):

    # Stats de ImageNet para ResNet50 preentrenada (https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if augment:
        # Train: mismas transformaciones base + augmentation geométrica y de flips.
        transform = T.Compose([
            T.ToPILImage(),                      # numpy array → PIL
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomAffine(
                degrees=15,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=5
            ),
            T.ToTensor(),                        # [0,255] → [0,1]
            T.Normalize(mean=mean, std=std),
        ])
    else:
        # Evaluación: solo redimensionar y normalizar, sin augmentation.

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    return transform