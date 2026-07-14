# renal-calculi-classificaction

Trabajo de Fin de Grado (TFG) — Grado en Ingeniería Informática, Escola Politècnica Superior, Universitat de les Illes Balears (UIB).

Este repositorio contiene el código del TFG *Estudis exploratoris LAIA@UIB: aplicació de models d'intel·ligència artificial per a la classificació d'imatges de càlculs renals*, que aborda la clasificación automática del tipo de cálculo renal a partir de imágenes, comparando modelos de aprendizaje superficial y profundo.

## Descripción del proyecto

El reconocimiento del tipo de cálculo renal es clave para prescribir un tratamiento adecuado y prevenir la recurrencia de la litiasis, pero el análisis morfo-constitucional que se emplea actualmente es lento y laborioso. Este trabajo explora en qué medida los modelos de inteligencia artificial pueden automatizar esa tarea a partir de imágenes.

Se emplea un conjunto de **397 imágenes *ex-vivo*** del Biobanco de Cálculos Renales de la UIB (BICUIB), clasificadas según la **taxonomía de Grases** en cinco clases: oxalato cálcico monohidrato (COM), oxalato cálcico dihidratado (COD), mixto, infeccioso y úrico.

El estudio compara tres clasificadores sobre dos tipos de entrada (imágenes completas y parches) y dos tareas (binaria y multiclase):

- **XGBoost** — aprendizaje superficial sobre descriptores manuales de textura (LBP) y color (HSV).
- **ResNet50** — red neuronal profunda preentrenada (transfer learning).
- **HybridResNet50** — modelo híbrido que fusiona las representaciones profundas de ResNet50 con los descriptores manuales.

La segmentación previa de las imágenes se realiza con **MedSAM**, y las cajas delimitadoras se anotan con Label Studio. La evaluación se hace mediante **validación cruzada `StratifiedGroupKFold` (K=5) agrupada por `stone_id`**, lo que evita que parches de un mismo cálculo se repartan entre entrenamiento y prueba. Para gestionar el desbalance se evalúan varias técnicas (ponderación de muestras, SMOTE, aumentación de datos y *weighted cross-entropy*).

**Hallazgo principal:** la limitación dominante no reside en el desbalance de clases, sino en la escasa separabilidad de las clases en el espacio de características —especialmente en el núcleo COM–COD–Mixto—, motivo por el cual las técnicas de balanceo evaluadas no mejoran el rendimiento.

## Tareas y escenarios

| Tarea | Clases |
|---|---|
| Binaria | COM vs. resto |
| Multiclase | COM, COD, Mixto, Infeccioso, Úrico |

| Entrada | Descripción |
|---|---|
| Imágenes completas | La imagen del cálculo con el fondo enmascarado |
| Parches | Fragmentos de 320×320 extraídos de la región del cálculo, agregados a nivel de imagen mediante *soft voting* |

## Organización del repositorio

> La estructura siguiente refleja la organización de los módulos. Ajusta los nombres de carpeta si en tu copia local difieren.

```
renal-calculi-classificaction/
├── src/
│   ├── data/
│   │   ├── dataset.py              # Dataset de PyTorch (KidneyStoneDataset) y transformaciones
│   │   ├── dataset_builder.py      # Construcción del DataFrame del dataset y etiquetado por tarea
│   │   ├── preprocessment.py       # TIFF→PNG, normalización de nombres, redimensionado, stone_id
│   │   ├── diagnostics.py          # Diagnóstico morfológico de las máscaras (cobertura, fragmentos)
│   │   ├── patch_extraction.py     # Extracción de parches centrados en la región del cálculo
│   │   └── patch_balancing.py      # Balanceo del conjunto de parches por sobremuestreo
│   ├── features/
│   │   ├── lbp.py                  # Descriptor de textura Local Binary Pattern
│   │   ├── hsv.py                  # Descriptores de color HSV (histograma y energía)
│   │   └── feature_extractor.py    # Combina LBP + HSV en un único vector por imagen
│   ├── models/
│   │   ├── resnet50.py             # ResNet50Classifier (backbone congelado + cabeza)
│   │   └── hybrid_resnet50.py      # HybridResNet50 (backbone + descriptores handcrafted)
│   ├── pipelines/
│   │   ├── xgboost_pipeline.py     # Experimento XGBoost (Optuna + CV agrupado)
│   │   ├── resnet50_pipeline.py    # Pipeline de entrenamiento/evaluación de ResNet50 por CV
│   │   └── hybrid_resnet50_pipeline.py  # Pipeline del modelo híbrido
│   ├── segmentation/
│   │   └── medsam_segmentor.py     # Segmentación con MedSAM a partir de cajas COCO
│   ├── analysis/
│   │   ├── embeddings.py           # Extracción de embeddings CNN + proyección UMAP
│   │   └── umap_generation.py      # Generación de las visualizaciones UMAP comparando modelos
│   ├── utils/
│   │   ├── io.py                   # Lectura de anotaciones COCO, carga de imágenes/máscaras
│   │   ├── checkpoints.py          # Guardado y carga de checkpoints
│   │   ├── constants.py            # Configuración base, rutas, enum Case, columnas
│   │   ├── reproducibility.py      # Semillas y generadores deterministas
│   │   └── visualization.py        # Matrices de confusión, gráficos de métricas, UMAP 2D
│   ├── train.py                    # Bucle de entrenamiento con early stopping
│   ├── predict.py                  # Inferencia sobre un DataLoader
│   └── evaluate.py                 # Cálculo, agregación y reporte de métricas
├── experiments/                    # Salidas: checkpoints, métricas, predicciones OOF, figuras
├── requirements.txt
└── README.md
```

### Flujo de datos

1. **Preprocesamiento** (`data/preprocessment.py`): conversión de formatos, normalización de nombres y redimensionado uniforme.
2. **Segmentación** (`segmentation/medsam_segmentor.py`): a partir de las cajas anotadas en Label Studio (formato COCO), MedSAM genera las máscaras binarias.
3. **Construcción del dataset** (`data/dataset_builder.py`): genera el DataFrame con rutas de imagen/máscara, etiqueta, vista y `stone_id`.
4. **Extracción de parches** (`data/patch_extraction.py`), opcional según el escenario.
5. **Extracción de características** (`features/`): LBP + HSV para XGBoost y para la rama manual del híbrido.
6. **Entrenamiento y evaluación** (`pipelines/`): CV `StratifiedGroupKFold` agrupada por `stone_id`.
7. **Análisis** (`analysis/`): proyecciones UMAP y matrices de confusión.

## Requisitos

- Python 3.10
- Las dependencias se listan en `requirements.txt`. A grandes rasgos: `torch`, `torchvision`, `xgboost`, `scikit-learn`, `imbalanced-learn` (SMOTE), `optuna`, `umap-learn`, `opencv-python`, `albumentations`, `segment-anything` (MedSAM), `pandas`, `numpy`, `matplotlib`, `seaborn`.

### Instalación

```bash
git clone https://github.com/Ti-trit/renal-calculi-classificaction.git
cd renal-calculi-classificaction

python3.10 -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

MedSAM requiere descargar el *checkpoint* preentrenado (arquitectura ViT-B) por separado; consulta el repositorio oficial de MedSAM e indica la ruta del checkpoint al instanciar el segmentador.

## Ejecución de los experimentos

> Los fragmentos siguientes son **ilustrativos** y muestran cómo se usan las clases principales. Adáptalos a tu punto de entrada real (script `main`, notebook o similar) y a las rutas de tu `constants.py`.

### 1. Segmentación

```python
from src.segmentation.medsam_segmentor import MedSAMSegmentor
from src.utils.io import load_coco_annotations, find_image_path, load_image

segmentor = MedSAMSegmentor(checkpoint="ruta/al/medsam_vit_b.pth")
images, ann_by_image = load_coco_annotations("anotaciones_coco.json")
# Para cada imagen: mask = segmentor.segment_image(image_rgb, anns)
```

### 2. Construcción del dataset

```python
from src.data.dataset_builder import DatasetBuilder

builder = DatasetBuilder(root_path="ruta/al/dataset")
df = builder.build(classes=[["1aI", "2a"], ["3aII", "3bII"], ...])  # agrupación por clase
builder.save_to_csv(df, "dataset.csv")
```

### 3. Experimento con XGBoost

```python
from src.pipelines.xgboost_pipeline import XGBoostExperiment
from src.features.feature_extractor import FeatureExtractor
from src.features.hsv import HSV_extractor
from src.features.lbp import LocalBinaryPattern
from src.utils.constants import Case

extractor = FeatureExtractor(hsv=HSV_extractor(), lbp=LocalBinaryPattern())
exp = XGBoostExperiment(df=df, extractor=extractor, case=Case.IMAGE)

# Búsqueda de hiperparámetros (Optuna) y evaluación por CV
search = exp.search_hyperparameters(config, use_sample_weight=False)
exp.evaluate_cv(config, best_params=search["best_params"])
```

### 4. Experimento con ResNet50

```python
from src.pipelines.resnet50_pipeline import resnet50
from src.utils.constants import Case

pipeline = resnet50(df=df, case=Case.IMAGE)
pipeline.run_model(config, use_weighted_cross_entropy=True)  # multiclase
```

### 5. Experimento con el modelo híbrido

```python
from src.pipelines.hybrid_resnet50_pipeline import HybridResNet50_pipeline
from src.utils.constants import Case

pipeline = HybridResNet50_pipeline(df=df, case=Case.IMAGE, extractor=extractor)
pipeline.run_model(config, use_weighted_cross_entropy=True)
```

### 6. Visualización UMAP

```python
from src.analysis.umap_generation import generate_umap
from src.utils.constants import Case

generate_umap(df, class_names=class_names, extractor=extractor,
              task_name="multiclase", case=Case.IMAGE)
```

Para el escenario de **parches**, se usa `Case.PATCH` en lugar de `Case.IMAGE` en cualquiera de los pipelines anteriores; los resultados a nivel de parche se agregan automáticamente a nivel de imagen mediante promedio de probabilidades.

## Salidas

Cada experimento escribe en `experiments/` (o en el `output_dir` del `config`):

- `checkpoints/best.pt` y `last.pt` — pesos del modelo (solo modelos profundos).
- `fold_results.json` y `summary.json` — métricas por *fold* y su media ± desviación.
- `oof_predictions_*.csv` — predicciones *out-of-fold*.
- Matrices de confusión y, con `analysis/`, las proyecciones UMAP.

> **Nota:** los *checkpoints* (`.pt`) pueden ocupar cientos de MB. Conviene mantenerlos fuera del control de versiones (`.gitignore`) o gestionarlos con Git LFS.

## Reproducibilidad

La semilla global se fija en `utils/constants.py` y se propaga a los generadores de datos, a la partición de la validación cruzada y al muestreo de parches. La ausencia de agrupación por `stone_id` en la partición produciría fuga de datos e inflaría artificialmente las métricas, por lo que toda la validación se realiza con `StratifiedGroupKFold` agrupado por cálculo.

