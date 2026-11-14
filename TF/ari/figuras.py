import argparse
import random
import re
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from auxiliares.dataset import ler_anotacoes
from auxiliares.preprocessamento import (TAMANHO, aumentar, carregar_mirror,
                                         ler_imagem_rgb, matriz_recorte,
                                         recortar, transformar_pontos)

PADRAO_EPOCA = re.compile(
    r"\[(\d+)/(\d+)\]\s*\|\s*Loss do treino:\s*(\S+)\s*\|\s*NME Valida\S*:\s*(\S+)%"
    r"\s*\|\s*NME olhos:\s*(\S+)%"
)


def desenhar_pontos(ax, pts):
  ax.scatter(pts[:, 0], pts[:, 1], s=8, c="lime")
  ax.scatter(pts[60:68, 0], pts[60:68, 1], s=14, c="red")
  ax.scatter(pts[68:76, 0], pts[68:76, 1], s=14, c="blue")


def figuras_pre_processamento(faces, mirror, pasta_saida, n=3, seed=0, mostrar=False):
  pasta_saida = Path(pasta_saida)
  pasta_saida.mkdir(parents=True, exist_ok=True)

  random.seed(seed)
  indices = random.sample(range(len(faces)), n)
  caminhos = []

  for i in indices:
    face = faces[i]
    M = matriz_recorte(face["caixa"])
    recorte, pts_rec = recortar(face, M)

    for _ in range(20):
      recorte_aug, pts_aug, aplicados = aumentar(recorte, pts_rec, mirror)
      if aplicados:
        break

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(ler_imagem_rgb(face["imagem"]))
    x0, y0, x1, y1 = face["caixa"]
    ax1.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                    fill=False, edgecolor="lime", linewidth=2))
    Minv = cv2.invertAffineTransform(M)
    cantos = np.array([[0, 0], [TAMANHO, 0], [TAMANHO, TAMANHO], [0, TAMANHO]], dtype=np.float32)
    ax1.add_patch(patches.Polygon(transformar_pontos(cantos, Minv), closed=True, fill=False,
                                  edgecolor="yellow", linestyle="--", linewidth=2))
    lm = face["landmarks"]
    ax1.scatter(lm[:, 0], lm[:, 1], s=6, c="red")
    ax1.set_title("imagem original (verde = caixa, amarelo = recorte)")
    ax1.axis("off")

    ax2.imshow(recorte)
    desenhar_pontos(ax2, pts_rec)
    ax2.set_title(f"recorte da face {TAMANHO}x{TAMANHO}")
    ax2.axis("off")

    ax3.imshow(recorte_aug)
    desenhar_pontos(ax3, pts_aug)
    ax3.set_title("aumento: " + (", ".join(aplicados) if aplicados else "nenhum"))
    ax3.axis("off")

    fig.tight_layout()
    caminho = pasta_saida / f"pre_processamento_face_{i}.png"
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    caminhos.append(caminho)
    if not mostrar:
      plt.close(fig)

  if mostrar:
    plt.show()
  return caminhos


def ler_log_treino(caminho_log):
  epocas, loss, nme, nme_olhos = [], [], [], []
  texto = Path(caminho_log).read_text(encoding="utf-8", errors="replace")
  for linha in texto.splitlines():
    m = PADRAO_EPOCA.search(linha)
    if m:
      epocas.append(int(m.group(1)))
      loss.append(float(m.group(3)))
      nme.append(float(m.group(4)))
      nme_olhos.append(float(m.group(5)))
  if not epocas:
    raise ValueError(
        f"nenhuma linha de época encontrada em {caminho_log}; salve a saída do treino com"
        " 'python3 -u train.py ... | tee treino.txt'"
    )
  return np.array(epocas), np.array(loss), np.array(nme), np.array(nme_olhos)


def grafico_treino(caminho_log, pasta_saida, mostrar=False):
  pasta_saida = Path(pasta_saida)
  pasta_saida.mkdir(parents=True, exist_ok=True)

  epocas, loss, nme, nme_olhos = ler_log_treino(caminho_log)
  melhor = int(np.argmin(nme + nme_olhos))

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

  ax1.plot(epocas, loss, color="tab:blue")
  ax1.set_xlabel("época")
  ax1.set_ylabel("loss do treino")
  ax1.set_title("Loss por época")
  ax1.grid(alpha=0.3)

  ax2.plot(epocas, nme, label="NME global", color="tab:orange")
  ax2.plot(epocas, nme_olhos, label="NME dos olhos", color="tab:green")
  ax2.axvline(epocas[melhor], color="gray", linestyle="--",
              label=f"melhor época ({epocas[melhor]})")
  ax2.set_xlabel("época")
  ax2.set_ylabel("NME na validação (%)")
  ax2.set_title("NME por época")
  ax2.grid(alpha=0.3)
  ax2.legend()

  fig.tight_layout()
  caminho = pasta_saida / "curvas_treino.png"
  fig.savefig(caminho, dpi=200, bbox_inches="tight")
  if mostrar:
    plt.show()
  else:
    plt.close(fig)
  return caminho


def main():
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest="comando", required=True)

  p1 = sub.add_parser("pre", help="figuras do pré-processamento: imagem, recorte e aumento")
  p1.add_argument("--data", required=True, help="pasta com WFLW_images e WFLW_annotations")
  p1.add_argument("--out", required=True, help="pasta onde salvar as figuras")
  p1.add_argument("--n", type=int, default=3, help="quantas figuras gerar")
  p1.add_argument("--seed", type=int, default=0)
  p1.add_argument("--mostrar", action="store_true", help="também abre as figuras na tela")

  p2 = sub.add_parser("treino", help="gráfico da loss e do NME por época")
  p2.add_argument("--log", required=True, help="arquivo com a saída do train.py")
  p2.add_argument("--out", required=True, help="pasta onde salvar o gráfico")
  p2.add_argument("--mostrar", action="store_true", help="também abre o gráfico na tela")

  args = ap.parse_args()

  if args.comando == "pre":
    faces = ler_anotacoes(args.data, "train")
    mirror = carregar_mirror(args.data)
    for caminho in figuras_pre_processamento(faces, mirror, args.out, args.n, args.seed, args.mostrar):
      print(f"salvo: {caminho}")
  else:
    print(f"salvo: {grafico_treino(args.log, args.out, args.mostrar)}")


if __name__ == "__main__":
  main()