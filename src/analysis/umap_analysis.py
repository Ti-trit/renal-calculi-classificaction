"""
Generación de las visualizaciones UMAP comparando los tres modelos.

Reúne, para una tarea y un caso (imagen o patch) dados, los embeddings de los
tres modelos —XGBoost (features LBP+HSV), ResNet50 e híbrido— y los proyecta con
UMAP en un mismo gráfico. La resolución de nombres, rutas de experimento y
checkpoints se hace mediante diccionarios de consulta, y cada modelo tiene su
extractor de embeddings registrado en ``EXTRACTORS``, de modo que
``generate_umap`` solo itera sobre ese registro.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.analysis.embeddings import extract_cnn_embeddings
from src.data.dataset import KidneyStoneDataset, get_transform
from src.pipelines.hybrid_resnet50_pipeline import HybridResNet50_pipeline, make_match_key
from src.pipelines.resnet50_pipeline import resnet50
from src.pipelines.xgboost_pipeline import XGBoostExperiment
from src.utils.checkpoints import load_checkpoint, load_hybrid_checkpoint
from src.utils.constants import Case, RESNET50_IMAGE_SIZE, BASE_CONFIG, NUM_WORKERS
from src.utils.visualization import plot_models_umap_2d


# Ajustes dependientes del caso: sufijo del experimento, etiqueta y subcarpeta.
CASE_CONFIG = {
    Case.IMAGE: {"suffix": "", "tipo": "imágenes", "folder": "images"},
    Case.PATCH: {"suffix": "_no_augmentation", "tipo": "parches", "folder": "patches"},
}

TASK_TITLES = {
    "binario": "Clasificación binaria",
    "multiclase": "Clasificación multiclase"
}

MODEL_LABELS = {
    "xgboost": "LBP+HSV",
    "resnet50": "ResNet50",
    "hybrid_resnet50": "Hybrid ResNet50",
}

DEFAULT_MODELS = ("xgboost", "resnet50", "hybrid_resnet50")




def experiment_dir(model_key, task_name, case):
    # Reconstruye nombre y ruta del experimento a partir de los diccionarios.
    cfg = CASE_CONFIG[case]
    name = f"{model_key}_{task_name}{cfg['suffix']}"
    return name, Path(f"experiments/{model_key}/{model_key}_{cfg['folder']}/{name}")


def build_config(model_key, task_name, case, class_names):
    name, output_dir = experiment_dir(model_key, task_name, case)
    return {**BASE_CONFIG, "name": name, "output_dir": str(output_dir),
            "class_names": class_names}


def checkpoint_path(model_key, task_name, case, fold):
    # Ruta al best.pt del fold indicado
    _, output_dir = experiment_dir(model_key, task_name, case)
    ckpt = output_dir / f"fold_{fold}" / "checkpoints" / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {ckpt}")
    return str(ckpt)


def make_loader(dataset, config):
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def emb_xgboost(df, case, extractor, **_):
    # Para XGBoost el "embedding" son directamente las features handcrafted.
    exp = XGBoostExperiment(df=df, extractor=extractor, case=case)
    return exp.features, exp.labels


def emb_resnet50(df, case, task_name, class_names, num_classes, device, fold, **_):
    """Reconstruye el modelo, recarga el checkpoint y extrae los embeddings CNN."""
    config = build_config("resnet50", task_name, case, class_names)
    model = resnet50(df=df, case=case).build_model(config, num_classes=num_classes)
    load_checkpoint(model, checkpoint_path("resnet50", task_name, case, fold), device)

    ds = KidneyStoneDataset(df, transform=get_transform(RESNET50_IMAGE_SIZE, augment=False), case=case)
    return extract_cnn_embeddings(model, make_loader(ds, config), device, num_classes=num_classes)


def emb_hybrid_resnet50(df, case, task_name, class_names, num_classes, device, fold, extractor, **_):
    config = build_config("hybrid_resnet50", task_name, case, class_names)
    exp = HybridResNet50_pipeline(df=df, case=case, extractor=extractor, xgboost_df=None)
    model = exp.build_model(config, num_classes=num_classes)
    load_hybrid_checkpoint(model, checkpoint_path("hybrid_resnet50", task_name, case, fold), device)

    # El híbrido necesita el match_key para emparejar cada imagen con su vector.
    df_with_key = df.assign(match_key=df[case.image_col].apply(make_match_key))
    ds = KidneyStoneDataset(
        df_with_key,
        transform=get_transform(RESNET50_IMAGE_SIZE, augment=False),
        case=case,
        handcrafted=exp.handcrafted,
        key_col="match_key",
    )
    return extract_cnn_embeddings(model, make_loader(ds, config), device, num_classes=num_classes)


# Registro modelo -> función extractora de embeddings.
EXTRACTORS = {
    "xgboost": emb_xgboost,
    "resnet50": emb_resnet50,
    "hybrid_resnet50": emb_hybrid_resnet50,
}

def generate_umap(df, class_names, extractor, task_name="multiclase",
                  case=Case.IMAGE, models=DEFAULT_MODELS, fold=0):

    num_classes = len(class_names)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Argumentos comunes; cada extractor toma los que necesita (resto vía **_).
    kwargs = dict(df=df, case=case, task_name=task_name, class_names=class_names,
                  num_classes=num_classes, device=device, fold=fold, extractor=extractor)

    # Embeddings de cada modelo, etiquetados con su nombre legible.
    features_dict = {}
    for key in models:
        emb, labels = EXTRACTORS[key](**kwargs)
        features_dict[MODEL_LABELS[key]] = (emb, labels)

    save_path = Path("experiments/plots/umap") / task_name / case.name / "umap_modelos_2d.pdf"
    return plot_models_umap_2d(
        features_dict,
        class_names=class_names,
        suptitle=f"Distribución de clases por modelo: {TASK_TITLES[task_name]} de {CASE_CONFIG[case]['tipo']}",
        save_path=save_path,
    )