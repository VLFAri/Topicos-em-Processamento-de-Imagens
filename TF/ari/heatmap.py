import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from auxiliares.dataset import ler_anotacoes
from auxiliares.preprocessamento import TAMANHO, recortar


def calcular_heatmaps(pts, tam_recorte=TAMANHO, tam_mapa=64, sigma=2.0):
  passo = tam_recorte / tam_mapa
  grade = np.arange(tam_mapa, dtype=np.float32)
  cx = pts[:, 0] / passo
  cy = pts[:, 1] / passo
  gx = np.exp(-((grade[None, :] - cx[:, None]) ** 2) / (2 * sigma**2))
  gy = np.exp(-((grade[None, :] - cy[:, None]) ** 2) / (2 * sigma**2))
  return gy[:, :, None] * gx[:, None, :]


def figura_heatmaps(face, tam_mapa, sigma, landmark):
  recorte, pts = recortar(face)
  mapas = calcular_heatmaps(pts, TAMANHO, tam_mapa, sigma)
  passo = TAMANHO / tam_mapa

  fig, eixos = plt.subplots(1, 4, figsize=(16, 4.5))

  ax = eixos[0]
  ax.imshow(recorte)
  ax.scatter(pts[:, 0], pts[:, 1], s=6, c="lime")
  ax.scatter(pts[landmark, 0], pts[landmark, 1], s=40, c="red")
  ax.text(pts[landmark, 0] - 4, pts[landmark, 1] - 6, str(landmark),
          color="red", fontsize=9, ha="right")
  ax.set_title(f"recorte {TAMANHO}x{TAMANHO}")

  ax = eixos[1]
  ax.imshow(mapas[landmark], cmap="magma", vmin=0, vmax=1)
  ax.set_title(f"H{landmark}: {tam_mapa}x{tam_mapa}, σ = {sigma:g}")

  ax = eixos[2]
  maximo = cv2.resize(mapas.max(axis=0), (TAMANHO, TAMANHO), interpolation=cv2.INTER_LINEAR)
  ax.imshow(recorte)
  ax.imshow(maximo, cmap="magma", alpha=0.6, vmin=0, vmax=1)
  ax.set_title(f"máximo dos {len(mapas)} mapas")

  ax = eixos[3]
  olho = pts[60:68]
  x0, y0 = olho.min(axis=0) - 12
  x1, y1 = olho.max(axis=0) + 12
  ax.imshow(recorte)
  ax.set_xlim(x0, x1)
  ax.set_ylim(y1, y0)
  ax.set_xticks(np.arange(-0.5, TAMANHO, passo))
  ax.set_yticks(np.arange(-0.5, TAMANHO, passo))
  ax.grid(True, color="white", linewidth=0.4, alpha=0.6)
  ax.tick_params(labelbottom=False, labelleft=False, length=0)
  ax.scatter(pts[[62, 66], 0], pts[[62, 66], 1], s=30, c="cyan")
  for k in (62, 66):
    ax.text(pts[k, 0] + 1, pts[k, 1] - 1, str(k), color="cyan", fontsize=9)
  distancia = np.linalg.norm(pts[62] - pts[66]) / passo
  ax.set_title(f"olho direito: grade {tam_mapa}x{tam_mapa} | 62-66: {distancia:.1f} células")

  for ax in eixos[:3]:
    ax.axis("off")
  fig.tight_layout()
  return fig


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--data", required=True, help="pasta com WFLW_images e WFLW_annotations")
  ap.add_argument("--out", required=True, help="pasta onde salvar as figuras")
  ap.add_argument("--n", type=int, default=3, help="quantas faces de teste mostrar")
  ap.add_argument("--sigma", type=float, default=2.0, help="sigma da gaussiana, em células do mapa")
  ap.add_argument("--mapa", type=int, default=64, help="lado do heatmap, em células")
  ap.add_argument("--landmark", type=int, default=60, help="landmark mostrado no segundo painel")
  ap.add_argument("--seed", type=int, default=0)
  ap.add_argument("--mostrar", action="store_true", help="também abre as figuras na tela")
  args = ap.parse_args()

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)

  faces = ler_anotacoes(args.data, "test")
  indices = random.Random(args.seed).sample(range(len(faces)), args.n)

  for i in indices:
    fig = figura_heatmaps(faces[i], args.mapa, args.sigma, args.landmark)
    caminho = out / f"heatmaps_face_{i}.png"
    fig.savefig(caminho, dpi=200)
    print(f"salvo: {caminho}")
    if not args.mostrar:
      plt.close(fig)

  if args.mostrar:
    plt.show()


if __name__ == "__main__":
  main()