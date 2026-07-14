
from pathlib import Path
from enum import Enum

class Case(Enum):
    IMAGE = ("image_path", "mask_path")
    PATCH = ("patch_path", "patch_mask_path")
    
    def __init__(self, image_col: str, mask_col: str):
        self.image_col = image_col
        self.mask_col = mask_col
#global
RUNNING_ON_LOCAL = False  
SEED = 57
TEST_SIZE= 0.20
PATCH_SIZE= 320
VAL_FRACTION_OF_TRAIN= 0.15
N_SPLITS= 5

PATCH_CONFIG = [224,256,288,320]

        
#ResNet50
RESNET50_IMAGE_SIZE= 224
NUM_WORKERS=2
EPOCHS= 30
DROPOUT= 0.3
LEARNING_RATE= 1e-4

#XGBoost
MIN_COVERAGE= 0.9
IMAGE_SIZE_XGBOOST= (1372, 1792)
N_TRIALS= 100 #Optuna
#LBP
RADIUS= 2
NUMPOINTS= 8
#para hsv también
N_BINS= 10



#Configuración de rutas
if RUNNING_ON_LOCAL:
    PROJECT_ROOT = Path("c:/Users/h/OneDrive/Documents/tfg/github")
    DATASET_PATH = Path(
        "C:/Users/h/OneDrive/Documents/tfg/renal calculi classification/"
        "DATASET/SECTION_VIEW_DATASET/SECTION_SURFACE_DATASET"
    )
    EXTRA_SYS_PATH = None
else:
    PROJECT_ROOT = Path("/home/ikk/renal-calculi-classificaction")
    DATASET_PATH = PROJECT_ROOT / "DATASET" / "SECTION_SURFACE_DATASET"
    EXTRA_SYS_PATH = "/home/ikk/.local/lib/python3.10/site-packages"

DATASET_PATH_XGBOOST= PROJECT_ROOT / "DATASET" / "RESIZED"
#  Common paths 
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
PATCHES_DIR = EXPERIMENTS_DIR / "patches"

#  Dataset columns 
IMG_COL = "image_path"
MASK_COL = "mask_path"
STONE_ID_COL = "stone_id"
PATCH_COL = "patch_path"
LABEL_COL= "label"
CLASS_COL= "folder"

# Classification tasks 
TASKS = {
    "binario": {
        "classes": [
            ["1aI", "2a"],
            ["3aII", "3bII", "4II-4I", "6", "8aI - 8aII"],
        ],
        "class_names": ["COM", "Otros"],
    },
    "multiclase": {
        "classes": [
            ["1aI", "2a"],
            ["3aII"],
            ["4II-4I", "3bII"],
            ["6"],
            ["8aI - 8aII"],
        ],
        "class_names": ["COM", "COD", "Mixto", "Infeccioso", "Urico"],
    },
    "com_subtypes": {
        "classes": [
            ["1aI"],
            ["2a"],
        ],
        "class_names": ["COM_papillary", "COM_cavity"],
    },
}

TASK_CONFIG = {
        "binary": {
            "objective": "binary:logistic",
            "scoring": "roc_auc",
            "eval_metric": "logloss",
        },
        "multiclass": {
            "objective": "multi:softprob",
            "scoring": "f1_macro",
            "eval_metric": "mlogloss",
        },
}


XGB_PARAM_SPACE_OPTUNA = {
    "max_depth":        ("int", 3, 10),
    "min_child_weight": ("int", 1, 10),
    "n_estimators":     ("int", 100, 600),
    "learning_rate":    ("float_log", 0.01, 0.3),
    "subsample":        ("float", 0.5, 1.0),
    "colsample_bytree": ("float", 0.3, 1.0),
   
}

BASE_CONFIG = {
    "batch_size": 8,

}
