"""Utilidades para la reproducibilidad: control de semillas para Python, NumPy, PyTorch y sklearn.
"""
import os
import random
import numpy as np
import torch


def set_seed(seed: int , deterministic_cuda: bool = True) -> None:

    random.seed(seed)
    # NumPy 
    np.random.seed(seed)
    # PyTorch CPU +  GPUs
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Variable de entorno 
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
   
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: int ) -> torch.Generator:
    
    g = torch.Generator()
    g.manual_seed(seed)
    return g