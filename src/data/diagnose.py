
import cv2
from pathlib import Path
import pandas as pd
import numpy as np



def diagnose_image(mask_path):
    """Analiza una imagen calcula la área que ocupa (cobertura), su numero de fragmentos,
    y el tamaño del fragmento de mayor y menor tamaño"""
    mask = cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    #binarizar la máscara
    mask = (mask > 0).astype(np.uint8)


    stone_pixels = mask.sum()
    
    # Encontrar el bbox del cálculo
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    
    bbox_h = ys.max() - ys.min()+1
    bbox_w = xs.max() - xs.min()+1
    
    # Encontrar los componentes conexos (fragmentos)
    n_components, labels_img = cv2.connectedComponents(mask)
    n_fragments = n_components - 1   # restamos el fondo que también es considerado un componente
    
    # Tamaño de los fragmentos 
    component_sizes = [
        (labels_img == i).sum() for i in range(1, n_components)
    ]
    if component_sizes is None:
        component_sizes= [0]
 
    return {
        "image_shape": mask.shape,
        "stone_pixels": int(stone_pixels),
        "coverage": float(stone_pixels/(mask.shape[0]*mask.shape[1])),
        "bbox": (bbox_h, bbox_w),
        "n_fragments": n_fragments,
        "min_fragment_pixels": int(min(component_sizes)),
        "max_fragment_pixels": int(max(component_sizes)),
    }


# Aplicar a todas las imágenes
def apply_diagnostic(root_path,classes):
    root= Path(root_path)
    diagnostics= []
    for class_dir in root.iterdir():
        if not class_dir.is_dir():
                continue
        folder_name = class_dir.name
        if folder_name not in classes:
            continue

        for view_dir in class_dir.iterdir():
            if not view_dir.is_dir():
                continue

            for img_file in view_dir.iterdir():
                if img_file.is_dir():       # carpeta MASKS
                    continue
                if img_file.suffix.lower() not in ('.jpg', '.png'):
                    continue
                mask_path = view_dir / "MASKS" / (img_file.stem + "_mask.png")
                d = diagnose_image(mask_path)
                d["image"] = Path(img_file).name
                d["folder"]= folder_name
                d["view"]= view_dir.name
                diagnostics.append(d)
    return pd.DataFrame(diagnostics) 

   