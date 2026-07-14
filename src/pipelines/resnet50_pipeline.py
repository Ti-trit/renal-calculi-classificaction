"""
Pipeline de entrenamiento y evaluación de ResNet50 por validación cruzada.

Orquesta el experimento K-fold con StratifiedGroupKFold agrupado por
``stone_id``: para cada fold separa un val set interno respetando los grupos,
entrena con ``fit``, recarga el mejor checkpoint, predice sobre test y agrega
las métricas entre folds. Para el caso de patches añade la agregación a nivel de
imagen. Es la clase base de la que hereda el pipeline híbrido.
"""

import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold

from src.models.resnet50 import ResNet50Classifier
from src.train import fit
from src.predict import predict
from src.evaluate import *
from src.utils.checkpoints import load_checkpoint
from src.data.dataset import KidneyStoneDataset, get_transform
from src.utils.reproducibility import seed_worker, get_generator
from src.utils.constants import *
from src.utils.io import save_json, prepare_output_dir
from src.utils.visualization import plot_confusion_matrix
from sklearn.utils.class_weight import compute_class_weight


class resnet50:

    def __init__(self, df: pd.DataFrame, case: Case):
        self.df = df
        self.case = case
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run_model(self, config: dict, show_fold_metrics: bool = False, use_weighted_cross_entropy: bool = False):
        print(f"Use weighted cross entropy: {use_weighted_cross_entropy}")
        name = config["name"]
        output_dir = prepare_output_dir(config)
        class_names = config["class_names"]

        print(f"\nEXPERIMENTO K-FOLD: {name}")
        print(f"Total {self.case.name.lower()}s: {len(self.df)}")
        print(f"K-folds: {N_SPLITS} | Validación interna: {VAL_FRACTION_OF_TRAIN:.0%} del train")

        # Partición externa agrupada por cálculo (stone_id)
        skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        folds = list(skf.split(self.df, y=self.df[LABEL_COL], groups=self.df[STONE_ID_COL]))

        fold_results = []
        all_preds = []

        for fold_idx, (trainval_idx, test_idx) in enumerate(folds):
            print(f"\n--- Fold {fold_idx + 1}/{N_SPLITS} ---")

            df_trainval = self.df.iloc[trainval_idx].reset_index(drop=True)
            df_test = self.df.iloc[test_idx].reset_index(drop=True)

            # Del trainval se separa un val interno, también respetando grupos
            df_train, df_val = split_trainval_by_group(
                df_trainval,
                val_fraction=VAL_FRACTION_OF_TRAIN,
                seed=SEED + fold_idx,
            )

            fold_dir = output_dir / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            result = self.run_single_fold(
                df_train=df_train,
                df_val=df_val,
                df_test=df_test,
                fold_idx=fold_idx,
                fold_dir=fold_dir,
                config=config,
                class_names=class_names,
                show_fold_metrics=show_fold_metrics,
                use_weighted_cross_entropy=use_weighted_cross_entropy,
            )

            fold_results.append(result)
            all_preds.append(result["fold_test_df"])

        fold_results_flat = []
        for fr in fold_results:
            fold_metrics = dict(fr["fold_metrics"])  # copia
            fold_metrics["fold"] = fr["fold"] + 1  # 1-indexado como XGBoost
            fold_results_flat.append(fold_metrics)

        # Resumen agregado
        aggregated = aggregate_fold_metrics(fold_results_flat)

        save_json(fold_results_flat, output_dir / "fold_results.json")
        save_json(aggregated, output_dir / "summary.json")

        # Predicciones out-of-fold de todos los folds concatenadas
        all_preds_df = pd.concat(all_preds, ignore_index=True)
        all_preds_df.to_csv(output_dir / f"oof_predictions_{self.case.name}.csv", index=False)

        if self.case == Case.PATCH:
            print("\n Métricas agregadas a nivel imagen (media ± std entre folds):")
            image_fold_metrics, image_summary, y_true_img, y_pred_img = \
                compute_per_image_metrics_by_fold(
                    oof_df=all_preds_df,
                    fold_col="fold",
                    image_id_col="source_image",
                    class_names=class_names,
                )
            save_json(image_fold_metrics, output_dir / "per_image_fold_results.json")
            save_json(image_summary, output_dir / "per_image_summary.json")
            print_summary(image_summary)

            plot_confusion_matrix(
                y_true_img, y_pred_img, class_names, output_dir,
                title=f"Matriz de confusión agregada por imagen — {name}",
            )

        plot_confusion_matrix(
            all_preds_df["true"], all_preds_df["pred"],
            class_names, output_dir,
            title=f"Matriz de confusión — {name}",
        )

        print_summary(aggregated)

        return {
            "name": name,
            "fold_results": fold_results_flat,  # devolver el formato plano
            "summary": aggregated,
        }

    def run_single_fold(
            self,
            df_train: pd.DataFrame,
            df_val: pd.DataFrame,
            df_test: pd.DataFrame,
            fold_idx: int,
            fold_dir: Path,
            config: dict,
            class_names: list,
            show_fold_metrics: bool,
            use_weighted_cross_entropy: bool
    ) -> dict:
        """Entrena un fold y devuelve las métricas """
        print(f"  Train: {len(df_train)} {self.case.name.lower()}s ({df_train[STONE_ID_COL].nunique()} cálculos)")
        print(f"  Val:   {len(df_val)} {self.case.name.lower()}s ({df_val[STONE_ID_COL].nunique()} cálculos)")
        print(f"  Test:  {len(df_test)} {self.case.name.lower()}s ({df_test[STONE_ID_COL].nunique()} cálculos)")

        # Build datasets and loaders
        train_loader, val_loader, test_loader = self.build_loaders(
            df_train, df_val, df_test, config, fold_idx
        )
        # Pesos de clase balanceados si se pide (para la CrossEntropy)
        if use_weighted_cross_entropy:
            y_train = df_train[LABEL_COL].values
            class_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.unique(y_train),
                y=y_train,
            )
        else:
            class_weights = None
        # Build and train model
        model = self.build_model(config, num_classes=len(class_names))
        fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=self.device,
            output_dir=fold_dir,
            class_weights=class_weights
        )

        # cargar el mejor checkpoint y predicción sobre test
        # fit deja en memoria la última época; se recarga best.pt para evaluar la
        # mejor época (carga in-place sobre `model`).
        load_checkpoint(model, str(fold_dir / "checkpoints" / "best.pt"), self.device)
        preds, labels, probs = predict(model, test_loader, self.device, return_probs=True)
        # Evaluar
        fold_metrics, df_test = evaluate_predictions(
            df_test=df_test,
            preds=preds,
            labels=labels,
            probs=probs,
            class_names=class_names,
            fold_idx=fold_idx,
        )

        if show_fold_metrics:
            print(f"  Fold {fold_idx}: " + " | ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in fold_metrics.items()
                if k != "fold" and isinstance(v, (int, float))
            ))

        # Guargar las métricas y predicciones
        save_metrics({f"{self.case.name}": fold_metrics}, str(fold_dir))
        save_predictions(df_test, preds, str(fold_dir), probs)

        return {
            "fold": fold_idx,
            "fold_metrics": fold_metrics,
            "fold_test_df": df_test,
        }

    def build_loaders(
            self,
            df_train: pd.DataFrame,
            df_val: pd.DataFrame,
            df_test: pd.DataFrame,
            config: dict,
            fold_idx: int,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Construye los tres DataLoaders con semillas controladas."""
        batch_size = config["batch_size"]

        # Augmentation solo en train; val y test usan la transformación de eval.
        train_tf = get_transform(image_size=RESNET50_IMAGE_SIZE, augment=True)
        eval_tf = get_transform(image_size=RESNET50_IMAGE_SIZE, augment=False)

        def make_dataset(df, transform):
            return KidneyStoneDataset(
                df,
                transform=transform,
                case=self.case,
            )

        train_ds = make_dataset(df_train, train_tf)
        val_ds = make_dataset(df_val, eval_tf)
        test_ds = make_dataset(df_test, eval_tf)

        # generator por fold fija el orden de shuffle de forma reproducible;
        # val/test van sin shuffle, así que no necesitan generator.
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
            worker_init_fn=seed_worker, generator=get_generator(SEED + fold_idx),
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

    def build_model(self, config: dict, num_classes: int) -> torch.nn.Module:
        model = ResNet50Classifier(
            num_classes=num_classes,
        ).to(self.device)
        return model


def split_trainval_by_group(
        df: pd.DataFrame,
        val_fraction: float,
        seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa un val set respetando stone_id, usando un único fold de SGKF."""
    # n_splits deriva de val_fraction (p. ej. 0.2 -> 5 splits) y toma el primero.
    n_splits = max(int(round(1.0 / val_fraction)), 2)
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(skf.split(df, y=df[LABEL_COL], groups=df[STONE_ID_COL]))
    return (
        df.iloc[train_idx].reset_index(drop=True),
        df.iloc[val_idx].reset_index(drop=True),
    )


def evaluate_predictions(
        df_test: pd.DataFrame,
        preds: np.ndarray,
        labels: np.ndarray,
        probs: np.ndarray,
        class_names: list,
        fold_idx: int,
) -> tuple[dict, pd.DataFrame]:
    print(f"\n  [Fold {fold_idx + 1}] Métricas :")
    print_report(labels, preds, y_proba=probs, class_names=class_names)
    metricas_fold = compute_metrics(labels, preds, y_proba=probs, class_names=class_names)

    test_df = build_predictions_df(df_test, preds, labels, probs, class_names, fold_idx)

    return metricas_fold, test_df


def build_predictions_df(
        df_test: pd.DataFrame,
        preds: np.ndarray,
        labels: np.ndarray,
        probs: np.ndarray,
        class_names: list,
        fold_idx: int,
) -> pd.DataFrame:
    """Crea un DataFrame con predicciones, labels, y probabilidades por clase."""
    out = df_test.copy()
    out["pred"] = preds
    out["true"] = labels
    for c, cname in enumerate(class_names):
        out[f"prob_{cname}"] = probs[:, c]
    out["fold"] = fold_idx
    return out