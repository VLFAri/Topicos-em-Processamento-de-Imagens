"""Baixa o dataset SVHN (splits train e test) via torchvision.

Uso:
    python baixar_dados.py                  # só baixa os arquivos .mat
    python baixar_dados.py --salvar-png     # também exporta as imagens como PNG
    python baixar_dados.py --amostras       # salva uma grade com exemplos de cada split
"""

import argparse
from pathlib import Path

import numpy as np
from torchvision import datasets
from torchvision.datasets.utils import download_url

SPLITS = ["train", "test"]

# Espelho dos mesmos arquivos .mat, usado quando o servidor de Stanford está fora do ar.
# O MD5 é conferido com o valor oficial do torchvision, então os arquivos são idênticos.
ESPELHO = "https://huggingface.co/datasets/MarMaster/svhn/resolve/main/"


def carregar_split(raiz: Path, split: str) -> datasets.SVHN:
    try:
        return datasets.SVHN(root=str(raiz), split=split, download=True)
    except Exception as erro:
        print(f"[{split}] falha no servidor oficial ({erro}); usando espelho")
        _, arquivo, md5 = datasets.SVHN.split_list[split]
        download_url(ESPELHO + arquivo, str(raiz), arquivo, md5)
        return datasets.SVHN(root=str(raiz), split=split, download=False)


def baixar(raiz: Path) -> dict:
    raiz.mkdir(parents=True, exist_ok=True)
    conjuntos = {}
    for split in SPLITS:
        ds = carregar_split(raiz, split)
        conjuntos[split] = ds
    return conjuntos


def salvar_png(conjuntos: dict, destino: Path) -> None:
    # Estrutura: destino/<split>/<classe>/<indice>.png
    for split, ds in conjuntos.items():
        for c in range(10):
            (destino / split / str(c)).mkdir(parents=True, exist_ok=True)
        for i in range(len(ds)):
            img, rotulo = ds[i]
            img.save(destino / split / str(rotulo) / f"{i:06d}.png")
        print(f"[{split}] PNGs salvos em {destino / split}")


def salvar_amostras(conjuntos: dict, destino: Path, n: int = 32) -> None:
    import matplotlib.pyplot as plt  # só necessário para --amostras

    destino.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for split, ds in conjuntos.items():
        idx = rng.choice(len(ds), size=n, replace=False)
        fig, eixos = plt.subplots(4, n // 4, figsize=(n // 4 * 1.2, 4 * 1.4))
        for ax, i in zip(eixos.flat, idx):
            img, rotulo = ds[i]
            ax.imshow(img)
            ax.set_title(str(rotulo))
            ax.axis("off")
        fig.suptitle(f"SVHN - {split}")
        fig.tight_layout()
        caminho = destino / f"amostras_{split}.png"
        fig.savefig(caminho, dpi=100)
        plt.close(fig)
        print(f"[{split}] grade de amostras salva em {caminho}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download do dataset SVHN")
    parser.add_argument("--raiz", type=Path, default=Path("data"),
                        help="pasta onde os arquivos .mat serão salvos")
    parser.add_argument("--salvar-png", action="store_true",
                        help="exporta todas as imagens como PNG em <raiz>/png")
    parser.add_argument("--amostras", action="store_true",
                        help="salva uma grade com imagens de exemplo")
    args = parser.parse_args()

    conjuntos = baixar(args.raiz)
    if args.salvar_png:
        salvar_png(conjuntos, args.raiz / "png")
    if args.amostras:
        salvar_amostras(conjuntos, args.raiz / "amostras")


if __name__ == "__main__":
    main()
        