"""
Pipeline del modelo híbrido ResNet50 + descriptores handcrafted.

Extiende el pipeline de ResNet50 reutilizando toda su lógica de CV y solo
sobrescribe la construcción del modelo (``HybridResNet50``) y de los loaders,
para inyectar en cada muestra su vector handcrafted (LBP+HSV). El emparejamiento
imagen<->handcrafted se hace por ``match_key`` (últimas tres partes de la ruta),
con validaciones que garantizan una correspondencia 1:1 sin claves duplicadas ni
faltantes.
"""

from src.pipelines.resnet50_pipeline import resnet50

import pandas as pd
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from src.data.dataset import KidneyStoneDataset, get_transform
from src.utils.reproducibility import seed_worker, get_generator
from src.utils.constants import *
from src.models.hybrid_resnet50 import HybridResNet50


class HybridResNet50_pipeline(resnet50):
    def __init__(
            self,
            df: pd.DataFrame,
            case: Case,
            extractor,
            xgboost_df: pd.DataFrame = None
    ):
        super().__init__(df, case)
        self.case = case
        self.df = df.reset_index(drop=True)
        # Si no se pasa un df aparte para el handcrafted, se usa el mismo df

        if xgboost_df is None:
            self.xgboost_df = df.reset_index(drop=True)
        else:
            self.xgboost_df = xgboost_df.reset_index(drop=True)

        # Construir la clave de emparejamiento a partir del image_col
        self.df["match_key"] = self.df[case.image_col].apply(make_match_key)
        self.xgboost_df["match_key"] = self.xgboost_df[case.image_col].apply(make_match_key)

        # Validaciones
        # Ambos df deben cubrir exactamente el mismo conjunto de claves
        keys_df = set(self.df["match_key"])
        keys_xgb = set(self.xgboost_df["match_key"])
        if keys_df != keys_xgb:
            missing = keys_df - keys_xgb
            extra = keys_xgb - keys_df
            raise ValueError(
                f"Mismatch entre df y xgboost_df por match_key.\n"
                f"  Faltan en xgboost_df ({len(missing)}): {list(missing)[:5]}\n"
                f"  Sobran en xgboost_df ({len(extra)}): {list(extra)[:5]}"
            )

        # La clave debe ser única en xgboost_df para poder indexar sin ambigüedad
        if self.xgboost_df["match_key"].duplicated().any():
            dups = self.xgboost_df[self.xgboost_df["match_key"].duplicated()]
            raise ValueError(
                f"xgboost_df tiene {len(dups)} claves duplicadas. "
                f"Primeras: {dups['match_key'].head().tolist()}"
            )

        # Extraer handcrafted
        handcrafted_xgb, _ = extractor.extract_features(
            self.xgboost_df,
            image_column_name=case.image_col,
            mask_column_name=case.mask_col

        )

        # Construir diccionario clave
        # Mapa match_key -> vector handcrafted, que el Dataset consulta por muestra
        self.handcrafted = {
            key: handcrafted_xgb[i]
            for i, key in enumerate(self.xgboost_df["match_key"].values)
        }
        self.n_handcrafted = handcrafted_xgb.shape[1]

    def build_model(self, config, num_classes):
        # Modelo híbrido: necesita saber la dimensión del vector handcrafted
        return HybridResNet50(
            num_classes=num_classes,
            n_handcrafted=self.n_handcrafted,
        ).to(self.device)

    def build_loaders(self, df_train, df_val, df_test, config, fold_idx):
        batch_size = config["batch_size"]
        train_tf = get_transform(RESNET50_IMAGE_SIZE, augment=True)
        eval_tf = get_transform(RESNET50_IMAGE_SIZE, augment=False)

        def make_dataset(df_split, transform):
            # handcrafted + key_col hacen que el Dataset devuelva también el
            # vector handcrafted emparejado por match_key (batch imagen+features)
            return KidneyStoneDataset(
                df_split,
                transform=transform,
                case=self.case,
                handcrafted=self.handcrafted,
                key_col="match_key",
            )

        train_ds = make_dataset(df_train, train_tf)
        val_ds = make_dataset(df_val, eval_tf)
        test_ds = make_dataset(df_test, eval_tf)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
            worker_init_fn=seed_worker,
            generator=get_generator(SEED + fold_idx),
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
            worker_init_fn=seed_worker,
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=NUM_WORKERS, pin_memory=True,
            worker_init_fn=seed_worker,
        )
        return train_loader, val_loader, test_loader


def make_match_key(path: str) -> str:
    """
    Extrae las últimas tres partes de la ruta como clave identificativa:
    {clase}/{vista}/{nombre_archivo}
    """
    p = Path(path)
    return str(Path(*p.parts[-3:]))