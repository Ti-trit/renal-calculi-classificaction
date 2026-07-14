"""Project bootstrap. Run at the start of every notebook:

    %run bootstrap.py

"""
import os
import sys
from pathlib import Path

# 1. Project root = donde está este archivo
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

# 2. Añadir raíz al sys.path para que `src.X` se pueda importar
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 3. Importar las constantes del proyecto
from src.utils.constants import *

if EXTRA_SYS_PATH is not None and EXTRA_SYS_PATH not in sys.path:
    sys.path.insert(0, EXTRA_SYS_PATH)

from src.utils.reproducibility import set_seed, seed_worker, get_generator
set_seed(SEED)

print(f"[Bootstrap] cwd={os.getcwd()} | seed={SEED}")