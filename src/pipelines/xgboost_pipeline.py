"""
Experimento de clasificación con XGBoost sobre descriptores handcrafted.

Cubre el flujo completo del modelo clásico: extracción de features (LBP+HSV),
búsqueda de hiperparámetros con Optuna (TPE) bajo StratifiedGroupKFold, y
evaluación por validación cruzada agrupada por ``stone_id`` con opciones de
balanceo (sample_weight, SMOTE) y augmentación en el espacio de imagen. Para el
caso de patches agrega además las métricas a nivel de imagen. Todas las
particiones se agrupan por cálculo para evitar fugas de datos entre train y test.
"""

import optuna
import xgboost as xgb
from sklearn.model_selection import (
    StratifiedGroupKFold,

)

from src.evaluate import *
from src.utils.constants import *
from src.utils.io import save_json, prepare_output_dir
from sklearn.utils.class_weight import compute_sample_weight
from optuna.samplers import TPESampler
from sklearn.metrics import get_scorer
from imblearn.over_sampling import SMOTE
from src.utils.visualization import plot_confusion_matrix
import albumentations as A
import cv2

optuna.logging.set_verbosity(optuna.logging.WARNING)

from typing import Optional


class XGBoostExperiment:

    def __init__(
            self,
            df: pd.DataFrame,
            extractor,
            case: Case,
            use_energies: bool = False,
    ):
        self.df = df
        self.extractor = extractor
        self.case = case

        self.features, self.labels = self.extract_features(use_energies)
        self.groups = self.get_validated_groups()

        # Binario vs multiclase determina objetivo y scoring de XGBoost.
        self.is_binary = len(np.unique(self.labels)) == 2
        self.task_cfg = (
            TASK_CONFIG["binary"] if self.is_binary else TASK_CONFIG["multiclass"]
        )

    def extract_features(self, use_energies: bool) -> tuple[np.ndarray, np.ndarray]:
        features, labels = self.extractor.extract_features(
            self.df,
            image_column_name=self.case.image_col,
            mask_column_name=self.case.mask_col,
            use_energies=use_energies
        )
        print(f"  Features shape: {features.shape}")
        print(f"  Distribución: {dict(zip(*np.unique(labels, return_counts=True)))}")
        return features, labels

    def get_validated_groups(self, group_col: str = "stone_id") -> np.ndarray:
        groups = self.df[group_col].values
        if pd.isna(groups).any():
            raise ValueError(f"Muestras sin {group_col}")
        return groups

    def search_hyperparameters(
            self,
            config: dict,
            use_sample_weight: bool = False,
    ) -> dict:
        name = config["name"]
        output_dir = prepare_output_dir(config)

        print(f"\nBÚSQUEDA HP: {name}")
        print(f"  use_sample_weight = {use_sample_weight}")
        best_params, search_info = run_optuna(
            X_train=self.features,
            y_train=self.labels,
            groups_train=self.groups,
            objective_xgb=self.task_cfg["objective"],
            scoring=self.task_cfg["scoring"],
            use_sample_weight=use_sample_weight,
        )
        print(f"  CV score: {search_info['best_score']:.4f}")
        print(f"  Best params: {best_params}")

        save_json(config, output_dir / "config.json")
        save_json(best_params, output_dir / "best_params.json")
        save_json(search_info, output_dir / "search_info.json")

        return {
            "name": name,
            "best_params": best_params,
            "search_info": search_info,
            "output_dir": output_dir,
        }

    def evaluate_cv(
        self,
        config: dict,
        best_params: dict,
        show_fold_metrics: bool = False,
        use_sample_weight: bool = False,
        use_augmentation: bool = False,
        apply_smote: bool = False,
    ) -> dict:
        name = config["name"]
        output_dir = prepare_output_dir(config)
        class_names = config.get("class_names")

        print(f"\nEVALUACIÓN CV: {name} ({N_SPLITS} folds)")
        print(f"  use_augmentation = {use_augmentation}")

        kfold = StratifiedGroupKFold(
            n_splits=N_SPLITS, shuffle=True, random_state=SEED,
        )

        fold_results = []
        fold_predictions = []

        for fold_idx, (tr_idx, te_idx) in enumerate(
                kfold.split(self.features, self.labels, groups=self.groups), start=1
        ):
            X_tr, X_te = self.features[tr_idx], self.features[te_idx]
            y_tr, y_te = self.labels[tr_idx], self.labels[te_idx]

            # Augmentación aplicada solo al train del fold
            if use_augmentation:
                X_aug, y_aug = generate_augmented_features(
                    train_df=self.df.iloc[tr_idx],
                    extractor=self.extractor,
                    image_col=self.case.image_col,
                    mask_col=self.case.mask_col,
                    fold_seed=SEED + fold_idx,
                    max_multiplier=10,
                    verbose=config.get("verbose", False),
                )
                if len(X_aug) > 0:
                    X_tr = np.concatenate([X_tr, X_aug])
                    y_tr = np.concatenate([y_tr, y_aug])

            # Garantía explícita de no solapamiento de grupos entre train y test.
            assert not (set(self.groups[tr_idx]) & set(self.groups[te_idx])), \
                f"Leakage en fold {fold_idx}"

            model = build_xgb_model(best_params, self.task_cfg["objective"])

            if apply_smote:
                # k_neighbors debe ser < nº de muestras de la clase minoritaria;
                # se acota al rango [1, 5] para evitar que SMOTE falle.
                _, counts = np.unique(y_tr, return_counts=True)
                min_class_count = np.min(counts)
                smote = SMOTE(
                    k_neighbors=max(1, min(5, min_class_count - 1)),
                    random_state=SEED,
                )
                X_tr, y_tr = smote.fit_resample(X_tr, y_tr)

            if use_sample_weight:
                sw = compute_sample_weight(class_weight="balanced", y=y_tr)
                model.fit(X_tr, y_tr, sample_weight=sw)
            else:
                model.fit(X_tr, y_tr)

            # Predicciones y métricas del fold
            y_pred = model.predict(X_te)
            y_proba = model.predict_proba(X_te)
            fold_metrics = compute_metrics(
                y_te, y_pred, y_proba=y_proba, class_names=class_names,
            )
            fold_metrics["fold"] = fold_idx
            fold_results.append(fold_metrics)
            fold_predictions.append({
                "fold": fold_idx,
                "y_true": y_te,
                "y_pred": y_pred,
                "y_proba": y_proba,
                "test_indices": te_idx,
            })

            if show_fold_metrics:
                print(f"  Fold {fold_idx}: " + " | ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in fold_metrics.items()
                    if k != "fold" and isinstance(v, (int, float))
                ))

        # Resumen de folds
        summary = aggregate_fold_metrics(fold_results)
        print_summary(summary)

        # Matriz de confusión agregada
        total_y_true = np.concatenate([fp["y_true"] for fp in fold_predictions])
        total_y_pred = np.concatenate([fp["y_pred"] for fp in fold_predictions])
        plot_confusion_matrix(
            total_y_true, total_y_pred, class_names, output_dir,
            title=f"Matriz de confusión — {name}",
        )

        # Construir DataFrame OOF
        # Predicciones out-of-fold: cada muestra aparece una vez, predicha por el
        # fold en que cayó en test. Las columnas prob_ se nombran por clase.
        oof_records = []
        for pred_data in fold_predictions:
            te_idx_fold = pred_data["test_indices"]
            fold_df = self.df.iloc[te_idx_fold].copy()
            fold_df["pred"] = pred_data["y_pred"]
            fold_df["true"] = pred_data["y_true"]
            fold_df["fold"] = pred_data["fold"]
            for c in range(pred_data["y_proba"].shape[1]):
                cname = class_names[c] if class_names else str(c)
                fold_df[f"prob_{cname}"] = pred_data["y_proba"][:, c]
            oof_records.append(fold_df)

        oof_df = pd.concat(oof_records, ignore_index=True)
        oof_df.to_csv(output_dir / f"oof_predictions_{self.case.name}.csv", index=False)

        # Si trabajamos con patches, agregar también a nivel imagen
        if self.case == Case.PATCH:
            print("\n Métricas agregadas a nivel imagen (media ± std entre folds):")
            image_fold_metrics, image_summary, y_true_img, y_pred_img = \
                compute_per_image_metrics_by_fold(
                    oof_df=oof_df,
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

        save_json(fold_results, output_dir / "fold_results.json")
        save_json(summary, output_dir / "summary.json")
        save_json(config, output_dir / "config.json")
        save_json(
            {"best_params_used": best_params, "use_sample_weight": use_sample_weight},
            output_dir / "eval_info.json",
        )

        return {
            "name": name,
            "fold_results": fold_results,
            "summary": summary,
        }


def build_xgb_model(params: dict, objective: str) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        **params,
        objective=objective,
        random_state=SEED,
        verbosity=0,
    )


def evaluate_fold(model, X_test, y_test, class_names: Optional[list]) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    return compute_metrics(y_test, y_pred, y_proba=y_proba, class_names=class_names)


def suggest_params_from_space(trial, space: dict) -> dict:
    # Traduce el espacio de búsqueda (dict de tipo+rango) a sugerencias Optuna.
    params = {}
    for name, (kind, lo, hi) in space.items():
        match kind:
            case "int":
                params[name] = trial.suggest_int(name, lo, hi)
            case "float":
                params[name] = trial.suggest_float(name, lo, hi)
            case "float_log":
                params[name] = trial.suggest_float(name, lo, hi, log=True)

    return params


def run_optuna(
        X_train, y_train, groups_train,
        objective_xgb: str,
        scoring: str,
        use_sample_weight: bool = False,
):
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    scorer = get_scorer(scoring)

    def objective(trial):
        # Cada trial se evalúa con CV agrupado interno: la métrica devuelta es
        # la media entre folds, para que la búsqueda no sobreajuste a un split.
        params = suggest_params_from_space(trial, XGB_PARAM_SPACE_OPTUNA)
        fold_scores = []
        for tr_idx, va_idx in cv.split(X_train, y_train, groups=groups_train):
            X_tr, X_va = X_train[tr_idx], X_train[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]

            model = build_xgb_model(params, objective_xgb)
            if use_sample_weight:
                sw = compute_sample_weight(class_weight='balanced', y=y_tr)
                model.fit(X_tr, y_tr, sample_weight=sw)
            else:
                model.fit(X_tr, y_tr)

            fold_scores.append(scorer(model, X_va, y_va))
        return float(np.mean(fold_scores))

    sampler = TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    return study.best_params, {
        "best_score": float(study.best_value),
        "n_trials": N_TRIALS,
        "use_sample_weight": use_sample_weight,
        "sampler_seed": SEED,
    }


def get_augmentation_transform(seed: int):
    return A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=8,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.5,
        ),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussNoise(std_range=(0.05, 0.15), p=0.3),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ], seed=seed)


def compute_augmentations_per_sample(
        class_counts: dict[str, int],
        max_multiplier: int = 3,
) -> dict[str, int]:
    """
    Calcula cuántas augmentaciones por muestra son necesarias para
    cada clase, con el objetivo de balancearla hacia la mayoritaria.

    Args:
        class_counts: dict {nombre_clase: n_muestras_en_train_fold}
        max_multiplier: tope máximo de augmentaciones por muestra


    Returns:
        dict {nombre_clase: n_aug_por_muestra}
    """
    if not class_counts:
        return {}

    max_count = max(class_counts.values())
    n_aug_per_sample = {}

    for cls, count in class_counts.items():
        if count >= max_count or count == 0:
            n_aug_per_sample[cls] = 0
            continue

        # Cuántas augmentaciones por muestra para alcanzar max_count
        needed_per_sample = (max_count - count) / count
        # Redondear y capar
        n_aug = min(int(round(needed_per_sample)), max_multiplier)
        n_aug_per_sample[cls] = n_aug

    return n_aug_per_sample


def generate_augmented_features(
        train_df: pd.DataFrame,
        extractor,
        image_col: str,
        mask_col: str,
        fold_seed: int,
        max_multiplier: int = 10,
        verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    # 1. Calcular distribución y plan de augmentación
    class_counts = train_df[CLASS_COL].value_counts().to_dict()
    n_aug_per_class = compute_augmentations_per_sample(
        class_counts, max_multiplier=max_multiplier,
    )

    if verbose:
        print(f"  Distribución train: {class_counts}")
        print(f"  Plan augmentación: {n_aug_per_class}")

    # 2. Aplicar augmentación a las muestras correspondientes
    aug_features, aug_labels = [], []

    for idx, (_, row) in enumerate(train_df.iterrows()):
        cls = row[CLASS_COL]
        n_aug = n_aug_per_class.get(cls, 0)
        if n_aug == 0:
            continue

        img_bgr = cv2.imdecode(
            np.fromfile(row[image_col], dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        # de BGR a RGB
        image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # mascara en escala de gris
        mask = cv2.imdecode(
            np.fromfile(row[mask_col], dtype=np.uint8),
            cv2.IMREAD_GRAYSCALE
        )

        for k in range(n_aug):
            # Semilla determinista y única por (fold, muestra, repetición): hace
            # la augmentación reproducible sin repetir la misma transformación.
            seed_k = fold_seed * 100000 + idx * 100 + k
            transform = get_augmentation_transform(seed=seed_k)

            result = transform(image=image, mask=mask)
            aug_image = result["image"]
            aug_mask = (result["mask"] > 0).astype(np.uint8) * 255

            feat = extractor.extract_image_features(aug_image, aug_mask, use_energies= False)
            aug_features.append(feat)
            aug_labels.append(row[LABEL_COL])

    if verbose:
        n_total_aug = len(aug_features)
        print(f"  Generadas {n_total_aug} muestras augmentadas")

    # Sin augmentaciones: devolver arrays vacíos con la forma esperada.
    if len(aug_features) == 0:
        return (
            np.array([]).reshape(0, extractor.feature_dim),
            np.array([]),
        )
    return np.array(aug_features), np.array(aug_labels)