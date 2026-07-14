"""
Construcción del DataFrame del dataset y asignación de etiquetas por tarea.

``DatasetBuilder`` recorre el árbol clase/vista y produce un DataFrame con las
rutas de imagen y máscara, la etiqueta, la carpeta de origen, la vista y el
``stone_id``. ``assign_task_labels`` reetiqueta un DataFrame de patches según la
agrupación de carpetas que define cada tarea (binaria o multiclase).
"""
from pathlib import Path
import pandas as pd

from src.data.preprocessment import extract_stone_id


class DatasetBuilder:

    def __init__(self, root_path: str, extensions=('.jpg', '.png')):
        self.root = Path(root_path)
        self.extensions = extensions

    def save_to_csv(self, df: pd.DataFrame, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    def build(
        self,
        classes: list

    ) -> pd.DataFrame:
        """
        Construye el DataFrame del dataset.

        Args:
            classes: lista de listas con los nombres de carpeta que componen cada clase.
            
        """
        # diccionario para asignar etiquetas a las carperas(subclace) que componen cada clase
        # Cada carpeta se mapea a (índice_de_clase, nombre_de_clase).
        folder_to_label = {}
        for class_idx, folder_list in enumerate(classes):
            class_name = f"classe_{class_idx}"
            for folder_name in folder_list:
                folder_to_label[folder_name] = (class_idx, class_name)

        records = []
        for class_dir in self.root.iterdir():
            if not class_dir.is_dir():
                continue
            folder_name = class_dir.name
            #carpetas no incluidas en la clasificación
            if folder_name not in folder_to_label:
                continue

            label, _ = folder_to_label[folder_name]
            #dos vistas: superfície y sección
            for view_dir in class_dir.iterdir():
                if not view_dir.is_dir():
                    continue

                for img_file in view_dir.iterdir():
                    if img_file.is_dir():       # carpeta MASKS
                        continue
                    if img_file.suffix.lower() not in self.extensions:
                        continue
                    #Máscara emparejada por nombre
                    mask_path = view_dir / "MASKS" / (img_file.stem + "_mask.png")
                    stone_id = extract_stone_id(img_file)

                    records.append({
                        "image_path": str(img_file),
                        "mask_path": str(mask_path) if mask_path.exists() else None,
                        "label": label,
                        "folder": folder_name,
                        "view": view_dir.name,
                        "stone_id": stone_id,
                    })

        df = pd.DataFrame(records)

        return df




     
def assign_task_labels(patches_df, task_config):
    """
    Devuelve una copia de patches_df con la columna 'label' asignada según la tarea.
    """
    # Mapeo folder -> índice de clase
    folder_to_label = {
        folder: idx
        for idx, folders in enumerate(task_config["classes"])
        for folder in folders
    }

    # Filtrar solo los folders que aplican a esta tarea
    df = patches_df[patches_df["folder"].isin(folder_to_label)].copy()
    df["label"] = df["folder"].map(folder_to_label).astype(int)

    return df.reset_index(drop=True)