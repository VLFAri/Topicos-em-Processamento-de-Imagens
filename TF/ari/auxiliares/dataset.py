from pathlib import Path
import numpy as np

ATRIBUTOS = ["pose", "expressão", "iluminação", "maquiagem", "oclusão", "desfoque"]

def ler_anotacoes(data_dir, split):
    data_dir = Path(data_dir)
    anotacoes = data_dir / "WFLW_annotations" / "list_98pt_rect_attr_train_test"

    faces = []
    for linha in (anotacoes / f"list_98pt_rect_attr_{split}.txt").read_text().splitlines():
        campos = linha.split()
        faces.append({
            "landmarks": np.array(campos[:196], dtype=np.float32).reshape(98, 2),
            "caixa": np.array(campos[196:200], dtype=np.float32),  # x_min y_min x_max y_max
            "atributos": np.array(campos[200:206], dtype=int),     # 0 = normal, 1 = difícil
            "imagem": data_dir / "WFLW_images" / campos[206],
        })
    return faces