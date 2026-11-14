import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from app import ear_medio
from auxiliares.dataset import ATRIBUTOS, ler_anotacoes
from auxiliares.preprocessamento import ler_imagem_rgb
from predict import LandmarkModel

REGIOES = {
    "contorno": (0, 33),
    "sobrancelhas": (33, 51),
    "nariz": (51, 60),
    "olho direito": (60, 68),
    "olho esquerdo": (68, 76),
    "boca": (76, 96),
    "pupilas": (96, 98),
}


def nme_por_face(pred: np.ndarray, alvo: np.ndarray):
  d = np.linalg.norm(alvo[60] - alvo[72])
  if d <= 0:
    return float("inf"), float("inf"), np.full(98, np.inf)
  erros_pts = np.linalg.norm(pred - alvo, axis=1) / d
  return float(erros_pts.mean()), float(erros_pts[60:76].mean()), erros_pts


def plotar_barras(
    rotulos,
    valores,
    titulo,
    rotulo_y,
    arquivo,
    out_dir: Path,
    cor="tab:blue",
    horizontal=False,
    linha_ref=None,
):
  plt.figure(figsize=(8, 5))
  if horizontal:
    barras = plt.barh(rotulos, valores, color=cor)
    plt.xlabel(rotulo_y)
    plt.gca().invert_yaxis()
    for barra, val in zip(barras, valores):
      plt.text(
          val + max(valores) * 0.015,
          barra.get_y() + barra.get_height() / 2,
          f"{val:.2f}%",
          va="center",
          fontsize=9,
      )
    plt.xlim(0, max(valores) * 1.15)
  else:
    barras = plt.bar(rotulos, valores, color=cor)
    plt.ylabel(rotulo_y)
    plt.xticks(rotation=20, ha="right")
    for barra, val in zip(barras, valores):
      plt.text(
          barra.get_x() + barra.get_width() / 2,
          val + max(valores) * 0.015,
          f"{val:.2f}%",
          ha="center",
          fontsize=9,
      )
    plt.ylim(0, max(valores) * 1.15)

  if linha_ref is not None:
    plt.axhline(
        y=linha_ref,
        color="red",
        linestyle="--",
        linewidth=1,
        label=f"Limiar de falha ({linha_ref}%)",
    )
    plt.legend()

  plt.title(titulo)
  plt.grid(axis="x" if horizontal else "y", alpha=0.3)
  plt.tight_layout()
  plt.savefig(out_dir / f"{arquivo}.png", dpi=200)
  plt.close()


def salvar_exemplos_qualitativos(exemplos, out_dir: Path):
  detectados = [e for e in exemplos if e["detectado"]]
  detectados.sort(key=lambda x: x["nme_global"])

  selecionados = detectados[:3] + detectados[-3:]
  titulos = [
      f"Acerto #{i + 1} (NME: {e['nme_global'] * 100:.2f}%)"
      for i, e in enumerate(detectados[:3])
  ] + [
      f"Falha #{i + 1} (NME: {e['nme_global'] * 100:.2f}%)"
      for i, e in enumerate(detectados[-3:])
  ]

  fig, axes = plt.subplots(2, 3, figsize=(14, 9))
  for ax, item, titulo in zip(axes.flat, selecionados, titulos):
    img = ler_imagem_rgb(item["imagem"]).copy()
    x0, y0, x1, y1 = item["caixa"]
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    lado = int(1.35 * max(w, h) / 2)

    H, W = img.shape[:2]
    rx0, ry0 = max(0, int(cx - lado)), max(0, int(cy - lado))
    rx1, ry1 = min(W, int(cx + lado)), min(H, int(cy + lado))

    recorte = img[ry0:ry1, rx0:rx1]
    ax.imshow(recorte)

    alvo = item["alvo"] - np.array([rx0, ry0])
    pred = item["pred"] - np.array([rx0, ry0])

    ax.scatter(alvo[:, 0], alvo[:, 1], s=10, c="lime", label="Anotado")
    ax.scatter(pred[:, 0], pred[:, 1], s=10, c="red", marker="x", label="Previsto")
    ax.set_title(titulo, fontsize=10)
    ax.axis("off")

  handles, labels = axes[0, 0].get_legend_handles_labels()
  fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)
  plt.tight_layout(rect=[0, 0, 1, 0.95])
  plt.savefig(out_dir / "exemplos_qualitativos.png", dpi=200)
  plt.close()


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--data", required=True, help="Caminho para o dataset WFLW")
  ap.add_argument("--weights", default="pesos/modelo.pt", help="Caminho dos pesos")
  ap.add_argument("--out", default="resultados_teste", help="Pasta de saída")
  args = ap.parse_args()

  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)

  device = "cuda" if torch.cuda.is_available() else "cpu"
  modelo = LandmarkModel(weights=args.weights, device=device)
  faces_teste = ler_anotacoes(args.data, "test")

  por_imagem = defaultdict(list)
  for idx, face in enumerate(faces_teste):
    por_imagem[str(face["imagem"])].append((idx, face))

  resultados = [None] * len(faces_teste)

  for caminho_img, lista_faces in por_imagem.items():
    img_rgb = ler_imagem_rgb(caminho_img)
    preds = modelo.predict(img_rgb)

    for idx, face in lista_faces:
      alvo = face["landmarks"]
      melhor_nme = float("inf")
      melhor_nme_olhos = float("inf")
      melhor_erros_pts = None
      melhor_pred = None

      for pred in preds:
        nme_g, nme_o, err_pts = nme_por_face(pred, alvo)
        if nme_g < melhor_nme:
          melhor_nme = nme_g
          melhor_nme_olhos = nme_o
          melhor_erros_pts = err_pts
          melhor_pred = pred

      detectado = melhor_pred is not None and melhor_nme < 0.50

      resultados[idx] = {
          "detectado": detectado,
          "nme_global": melhor_nme if detectado else float("nan"),
          "nme_olhos": melhor_nme_olhos if detectado else float("nan"),
          "erros_pts": melhor_erros_pts if detectado else None,
          "pred": melhor_pred,
          "alvo": alvo,
          "caixa": face["caixa"],
          "atributos": face["atributos"],
          "imagem": caminho_img,
          "ear_anotado": float(ear_medio(alvo)),
          "ear_previsto": float(ear_medio(melhor_pred)) if detectado else float("nan"),
      }

  total_faces = len(resultados)
  detectados = [r for r in resultados if r["detectado"]]
  n_det = len(detectados)
  taxa_deteccao = (n_det / total_faces) * 100.0

  nmes_g = np.array([r["nme_global"] for r in detectados]) * 100.0
  nmes_o = np.array([r["nme_olhos"] for r in detectados]) * 100.0

  falhas_g = sum(1 for r in resultados if not r["detectado"] or r["nme_global"] > 0.10)
  falhas_o = sum(1 for r in resultados if not r["detectado"] or r["nme_olhos"] > 0.10)

  taxa_falha_global = (falhas_g / total_faces) * 100.0
  taxa_falha_olhos = (falhas_o / total_faces) * 100.0

  nme_global_medio = float(nmes_g.mean())
  nme_olhos_medio = float(nmes_o.mean())

  # 1. Gráfico de resumo geral (Seção 12.1)
  plotar_barras(
      [
          "Taxa Detecção",
          "NME Global",
          "Falha Global (>10%)",
          "NME Olhos",
          "Falha Olhos (>10%)",
      ],
      [
          taxa_deteccao,
          nme_global_medio,
          taxa_falha_global,
          nme_olhos_medio,
          taxa_falha_olhos,
      ],
      "Métricas Gerais no Conjunto de Teste (WFLW)",
      "Percentual (%)",
      "resumo_metricas_teste",
      out_dir,
      cor=["tab:green", "tab:blue", "tab:red", "tab:cyan", "tab:orange"],
  )

  # 2. Gráfico NME por região da face (Seção 6)
  erros_matriz = np.stack([r["erros_pts"] for r in detectados]) * 100.0
  nme_regioes = {
      nome: float(erros_matriz[:, ini:fim].mean())
      for nome, (ini, fim) in REGIOES.items()
  }
  plotar_barras(
      list(nme_regioes.keys()),
      list(nme_regioes.values()),
      "NME por Região da Face no Conjunto de Teste",
      "NME (%)",
      "nme_por_regiao_teste",
      out_dir,
      cor="tab:purple",
      horizontal=True,
  )

  # 3. Gráfico NME por subconjunto de dificuldade (Seção 6)
  nme_subconjuntos = {}
  for k, nome in enumerate(ATRIBUTOS):
    vals = [
        r["nme_global"] * 100.0
        for r in detectados
        if r["atributos"][k] == 1
    ]
    nme_subconjuntos[nome] = float(np.mean(vals)) if vals else 0.0

  plotar_barras(
      list(nme_subconjuntos.keys()),
      list(nme_subconjuntos.values()),
      "NME por Subconjunto de Dificuldade no Teste (WFLW)",
      "NME (%)",
      "nme_por_subconjunto_teste",
      out_dir,
      cor="tab:orange",
      linha_ref=10.0,
  )

  # 4. Gráfico de dispersão EAR Anotado vs EAR Previsto (Seção 6)
  ear_real = np.array([r["ear_anotado"] for r in detectados])
  ear_pred = np.array([r["ear_previsto"] for r in detectados])
  correlacao = float(np.corrcoef(ear_real, ear_pred)[0, 1])
  eam_ear = float(np.mean(np.abs(ear_real - ear_pred)))

  plt.figure(figsize=(6, 6))
  plt.scatter(ear_real, ear_pred, alpha=0.35, s=12, color="tab:blue")
  lim = max(ear_real.max(), ear_pred.max()) * 1.05
  plt.plot([0, lim], [0, lim], "r--", label="Ideal (y = x)")
  plt.xlabel("EAR Anotado (Ground Truth)")
  plt.ylabel("EAR Previsto pelo Modelo")
  plt.title(f"EAR Anotado vs. Previsto (r = {correlacao:.3f} | EAM = {eam_ear:.4f})")
  plt.grid(alpha=0.3)
  plt.legend()
  plt.tight_layout()
  plt.savefig(out_dir / "dispersao_ear_teste.png", dpi=200)
  plt.close()

  # 5. Exemplos qualitativos (acertos e falhas)
  salvar_exemplos_qualitativos(resultados, out_dir)

  # Resumo no terminal
  print("=== RESULTADOS NO CONJUNTO DE TESTE (WFLW) ===")
  print(f"Faces detectadas: {n_det}/{total_faces} ({taxa_deteccao:.2f}%)")
  print(f"NME Global: {nme_global_medio:.2f}% | Taxa de falha (>10%): {taxa_falha_global:.2f}%")
  print(f"NME Olhos:  {nme_olhos_medio:.2f}% | Taxa de falha (>10%): {taxa_falha_olhos:.2f}%")
  print("\nNME por região da face:")
  for regiao, val in nme_regioes.items():
    print(f"  - {regiao}: {val:.2f}%")
  print("\nNME por subconjunto:")
  for sub, val in nme_subconjuntos.items():
    print(f"  - {sub}: {val:.2f}%")
  print(f"\nEAR (Teste) -> Correlação de Pearson: {correlacao:.4f} | EAM: {eam_ear:.4f}")
  print(f"\nGráficos salvos em: {out_dir.resolve()}")


if __name__ == "__main__":
  main()