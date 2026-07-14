"""
Bucle de entrenamiento para el clasificador basado en ResNet50.

Este módulo implementa el ciclo de entrenamiento/validación por época y la
función de alto nivel ``fit`` que orquesta el entrenamiento completo de un
fold: optimizador AdamW, scheduler ReduceLROnPlateau, early stopping sobre
``val_loss`` y guardado de checkpoints (mejor y último).

El diseño soporta con un mismo bucle tanto la ResNet50 pura como la variante
híbrida (imagen + características handcrafted) usando el desempaquetado
flexible de cada batch (``images, *extra, labels``).
"""

import time
from typing import Optional, Iterable
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from src.models.resnet50 import ResNet50Classifier
from src.utils.checkpoints import save_checkpoint
from src.utils.constants import LEARNING_RATE, EPOCHS
from src.utils.io import prepare_output_dir


def train_one_epoch(
    model: ResNet50Classifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> tuple:
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_preds, all_labels = [], []
     
    for batch in loader:
        images, *extra, labels= batch
        handcrafted = extra[0].to(device) if extra else None
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images, handcrafted)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
 

    avg_loss = total_loss / max(1, total_samples)
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
    return avg_loss, accuracy, f1


@torch.no_grad()
def validate_one_epoch(
    model: ResNet50Classifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds, all_labels = [], []

    for batch in loader:
        images, *extra, labels= batch
        handcrafted = extra[0].to(device) if extra else None
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images, handcrafted)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(1, total_samples)
    accuracy = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
    return avg_loss, accuracy, f1


def fit(
    model: ResNet50Classifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    class_weights: Iterable[float],
    weight_decay: float = 1e-4,
    output_dir: Optional[str] = None,
    save_best_metric: str = "val_loss",
    save_last: bool = True,
    early_stopping_patience: int= 10,
) -> pd.DataFrame:

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)

    # wighted cross entropy para clasificación multiclase
    if class_weights is not None:
        weights = torch.tensor(list(class_weights), dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizador
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=weight_decay)

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode= "min",patience=3,min_lr=1e-7)


    print(f"Device: {device}")
    print(f"Trainable parameters: {model.trainable_params():,}")
    print(f"Epochs: {EPOCHS}, LR: {LEARNING_RATE}, Weight decay: {weight_decay}")

    history = []
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
        )
        val_loss, val_acc, val_f1 = validate_one_epoch(
            model, val_loader, criterion, device,
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
            "lr": current_lr, "time_s": elapsed,
        }
        history.append(epoch_record)

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"train: loss={train_loss:.4f} acc={train_acc:.4f} f1={train_f1:.4f} | "
              f"val: loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} | "
              f"lr={current_lr:.2e} | {elapsed:.1f}s")

        # Mejor época?
        if val_loss < best_val_loss:
            best_val_loss= val_loss
            best_epoch= epoch
            epochs_without_improvement = 0
            if output_dir is not None:
                save_checkpoint(
                    model, output_dir / "checkpoints" / "best.pt", epoch_record,
                )
        else:
            epochs_without_improvement += 1

        if output_dir is not None and save_last:
            save_checkpoint(
                model, output_dir / "checkpoints" / "last.pt", epoch_record,
            )

        if output_dir is not None:
            pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

        if epochs_without_improvement >= early_stopping_patience:

            print(f"Early stopping en época {epoch}. Mejor: {best_epoch}.")
            print(f"Mejor val_loss: {best_val_loss:.4f} (época {best_epoch}).")

            break


    print(f"Mejor {save_best_metric}: {best_val_loss:.4f} (época {best_epoch})")

    return pd.DataFrame(history)