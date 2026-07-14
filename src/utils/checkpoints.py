"""
Guardado y carga de checkpoints de los modelos.

Persiste y restaura el ``state_dict`` de la red interna (``model.model``) junto
con el registro de la época. La carga se hace in-place sobre el modelo recibido
y devuelve el ``epoch_record`` asociado; también se ofrece un constructor que
reconstruye el modelo completo desde un checkpoint.
"""

from pathlib import Path
import torch
from src.models.resnet50 import ResNet50Classifier
from src.models.hybrid_resnet50 import HybridResNet50


def save_checkpoint(
        model: ResNet50Classifier,
        path: Path,
        epoch_record: dict,
) -> None:
    # Se guarda el state_dict de la red interna (model.model), no el wrapper.
    torch.save({
        "model_state_dict": model.model.state_dict(),
        "epoch_record": epoch_record
    }, path)


def load_checkpoint(
        model: ResNet50Classifier,
        path: str,
        device
) -> dict:
    # map_location=device permite cargar en CPU un checkpoint entrenado en GPU.
    ckpt = torch.load(path, map_location=device)
    # Carga in-place: sobrescribe los pesos del `model` recibido
    model.model.load_state_dict(ckpt["model_state_dict"])
    return ckpt.get("epoch_record", {})


def load_hybrid_checkpoint(
        model: HybridResNet50,
        path: str,
        device,
) -> dict:
    # Equivalente a load_checkpoint pero tipado para el modelo híbrido.
    ckpt = torch.load(path, map_location=device)
    model.model.load_state_dict(ckpt["model_state_dict"])
    return ckpt.get("epoch_record", {})


def load_model_from_checkpoint(path: str, device) -> ResNet50Classifier:
    # Reconstruye el modelo desde cero: el checkpoint debe incluir num_classes.
    ckpt = torch.load(path, map_location=device)
    model = ResNet50Classifier(
        num_classes=ckpt["num_classes"],
    )
    model.model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model