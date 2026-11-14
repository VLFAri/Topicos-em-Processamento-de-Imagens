import argparse
import json
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from auxiliares.dataset import ATRIBUTOS, ler_anotacoes
from auxiliares.preprocessamento import TAMANHO, aumentar, carregar_mirror, normalizar, recortar
from auxiliares.resnet18 import LandmarkResNet

seed = 0
fracao_val = 0.1
peso_olhos = 5.0
num_workers = 2

salvar_graficos = True
mostrar_graficos = False

regioes = {
    "contorno": (0, 33),
    "sobrancelhas": (33, 51),
    "nariz": (51, 60),
    "olho direito": (60, 68),
    "olho esquerdo": (68, 76),
    "boca": (76, 96),
    "pupilas": (96, 98),
}

cores = plt.cm.tab10.colors


class WeightedMSELoss(nn.Module):

  def __init__(self, eye_weight=5.0):
    super().__init__()
    self.mse = nn.MSELoss(reduction="none")
    self.eye_weight = eye_weight

  def forward(self, pred, target):
    loss = self.mse(pred, target)
    pesos = torch.ones(98, device=pred.device)
    pesos[60:76] = self.eye_weight
    pesos = pesos.view(1, 98, 1)
    return torch.mean(loss * pesos)


class WFLWDataset(Dataset):

  def __init__(self, faces, mirror, augment=True):
    self.faces = faces
    self.mirror = mirror
    self.augment = augment

  def __len__(self):
    return len(self.faces)

  def __getitem__(self, idx):
    recorte, pts = recortar(self.faces[idx])
    if self.augment:
      recorte, pts, _ = aumentar(recorte, pts, self.mirror)
    return {
        "imagem": torch.from_numpy(normalizar(recorte)),
        "landmarks": torch.from_numpy((pts / TAMANHO).astype(np.float32)),
    }


def fixar_semente(seed_val):
  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  torch.cuda.manual_seed_all(seed_val)


def dividir_por_imagem(faces, fracao_val_split, seed_val):
  imagens = sorted({str(f["imagem"]) for f in faces})
  random.Random(seed_val).shuffle(imagens)
  val_imgs = set(imagens[: int(len(imagens) * fracao_val_split)])
  treino = [f for f in faces if str(f["imagem"]) not in val_imgs]
  val = [f for f in faces if str(f["imagem"]) in val_imgs]
  return treino, val


def calcular_nmes(pred, alvo, atributos_val):
  d = np.linalg.norm(alvo[:, 60] - alvo[:, 72], axis=1)
  erro = np.linalg.norm(pred - alvo, axis=2) / d[:, None]
  nme_face = erro.mean(axis=1)

  metricas = {
      "global": float(nme_face.mean() * 100),
      "olhos": float(erro[:, 60:76].mean() * 100),
  }

  for nome, (inicio, fim) in regioes.items():
    metricas[f"regiao|{nome}"] = float(erro[:, inicio:fim].mean() * 100)

  for k, nome in enumerate(ATRIBUTOS):
    mascara = atributos_val[:, k] == 1
    metricas[f"atributo|{nome}"] = (
        float(nme_face[mascara].mean() * 100) if mascara.any() else float("nan")
    )

  return metricas


def nome_arquivo(texto):
  sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
  return sem_acento.lower().replace(" ", "_")


def gerar_graficos_agrupados(historico, contagem_atributos, pasta_out):
  if not salvar_graficos:
    return

  pasta_graficos = pasta_out / "graficos"
  pasta_graficos.mkdir(parents=True, exist_ok=True)

  e = historico["epoca"]

  # 1. Gráfico dedicado para a Loss do Treino
  plt.figure(figsize=(8, 5))
  plt.plot(e, historico["loss"], color="tab:blue", marker="o", markersize=3, label="Loss Treino")
  plt.xlabel("Épocas")
  plt.ylabel("Loss")
  plt.title("Evolução da Loss por Época")
  plt.grid(alpha=0.3)
  plt.legend()
  plt.tight_layout()
  plt.savefig(pasta_graficos / "evolucao_loss.png", dpi=200)
  plt.close()

  # 2. Gráfico 1: NME Global e NME dos Olhos (Validação)
  plt.figure(figsize=(9, 5))
  plt.plot(e, historico["global"], color="tab:orange", marker="o", markersize=3, label="NME Global (%)")
  plt.plot(e, historico["olhos"], color="tab:green", marker="s", markersize=3, label="NME Olhos (%)")
  plt.xlabel("Épocas")
  plt.ylabel("NME (%)")
  plt.title("Desempenho Geral: NME Global e NME dos Olhos por Época")
  plt.grid(alpha=0.3)
  plt.legend()
  plt.tight_layout()
  plt.savefig(pasta_graficos / "nme_geral_global_olhos.png", dpi=200)
  plt.close()

  # 3. Gráfico 2: NME por Região da Face
  plt.figure(figsize=(10, 6))
  for i, nome in enumerate(regioes):
    plt.plot(e, historico[f"regiao|{nome}"], color=cores[i % len(cores)], linewidth=1.5, label=f"Região: {nome}")
  plt.xlabel("Épocas")
  plt.ylabel("NME (%)")
  plt.title("Evolução da NME por Região da Face (Validação)")
  plt.grid(alpha=0.3)
  plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
  plt.tight_layout()
  plt.savefig(pasta_graficos / "nme_por_regiao_facial.png", dpi=200)
  plt.close()

  # 4. Gráfico 3: NME por Subconjunto de Atributos/Dificuldade
  plt.figure(figsize=(10, 6))
  for i, nome in enumerate(ATRIBUTOS):
    valores = historico[f"atributo|{nome}"]
    if np.isnan(valores).all():
      continue
    plt.plot(
        e,
        valores,
        color=cores[i % len(cores)],
        linewidth=1.5,
        label=f"{nome} (n={contagem_atributos[i]})",
    )
  plt.xlabel("Épocas")
  plt.ylabel("NME (%)")
  plt.title("Evolução da NME por Subconjunto de Dificuldade (WFLW)")
  plt.grid(alpha=0.3)
  plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
  plt.tight_layout()
  plt.savefig(pasta_graficos / "nme_por_subconjunto_wflw.png", dpi=200)
  plt.close()


def formatar_valor(valor):
  return "n/d" if np.isnan(valor) else f"{valor:.2f}%"


def formatar_epoca(epoca, total_epocas, epoca_loss, metricas):
  linha = (
      f"Época [{epoca}/{total_epocas}] | Loss do treino: {epoca_loss:.5f} |"
      f" NME Validação: {metricas['global']:.2f}% | NME olhos: {metricas['olhos']:.2f}%"
  )
  regioes_txt = " | ".join(
      f"{nome} {formatar_valor(metricas[f'regiao|{nome}'])}" for nome in regioes
  )
  atributos_txt = " | ".join(
      f"{nome} {formatar_valor(metricas[f'atributo|{nome}'])}" for nome in ATRIBUTOS
  )
  return f"{linha}\n  NME regiões: {regioes_txt}\n  NME subconjuntos: {atributos_txt}"


def imprimir_resumo(melhor_epoca, metricas):
  print(f"Melhor época: {melhor_epoca}")
  for chave, valor in metricas.items():
    if chave == "global":
      rotulo = "global"
    elif chave == "olhos":
      rotulo = "olhos (60 a 75)"
    else:
      tipo, nome = chave.split("|")
      rotulo = f"região {nome}" if tipo == "regiao" else f"subconjunto {nome}"
    print(f"  NME {rotulo}: {valor:.2f}%")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--data", required=True, help="Caminho para a pasta do dataset WFLW")
  ap.add_argument("--out", required=True, help="Caminho para salvar os resultados (pesos/logs)")
  ap.add_argument("--epochs", type=int, default=100, help="Número de épocas")
  ap.add_argument("--batch-size", type=int, default=64, help="Tamanho do batch")
  ap.add_argument("--lr", type=float, default=1e-4, help="Taxa de aprendizagem")
  args = ap.parse_args()

  out_dir = Path(args.out)
  out_dir.mkdir(parents=True, exist_ok=True)

  fixar_semente(seed)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  faces = ler_anotacoes(args.data, "train")
  mirror = carregar_mirror(args.data)
  treino, val = dividir_por_imagem(faces, fracao_val, seed)
  print(f"Treino: {len(treino)} faces | Validação: {len(val)} faces")

  atributos_val = np.stack([f["atributos"] for f in val])
  contagem_atributos = atributos_val.sum(axis=0)

  gerador = torch.Generator().manual_seed(seed)
  train_loader = DataLoader(
      WFLWDataset(treino, mirror, augment=True),
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=num_workers,
      generator=gerador,
  )
  val_loader = DataLoader(
      WFLWDataset(val, mirror, augment=False),
      batch_size=args.batch_size,
      shuffle=False,
      num_workers=num_workers,
  )

  model = LandmarkResNet(pretrained=True).to(device)
  total = sum(p.numel() for p in model.parameters())
  treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
  print(f"Parâmetros: {total:,} no total | {treinaveis:,} treináveis")

  criterion = WeightedMSELoss(eye_weight=peso_olhos)
  optimizer = optim.AdamW(model.parameters(), lr=args.lr)

  melhor = float("inf")
  melhor_epoca = 0
  melhores_metricas = {}
  historico = defaultdict(list)

  for epoca in range(args.epochs):
    model.train()
    running_loss = 0.0

    for batch in train_loader:
      imagens = batch["imagem"].to(device)
      landmarks_reais = batch["landmarks"].to(device)

      optimizer.zero_grad()
      saidas = model(imagens)
      loss = criterion(saidas, landmarks_reais)
      loss.backward()
      optimizer.step()

      running_loss += loss.item()

    epoca_loss = running_loss / len(train_loader)

    model.eval()
    preds, alvos = [], []

    with torch.no_grad():
      for batch in val_loader:
        saidas = model(batch["imagem"].to(device))
        preds.append(saidas.cpu().numpy())
        alvos.append(batch["landmarks"].numpy())

    metricas = calcular_nmes(
        np.concatenate(preds), np.concatenate(alvos), atributos_val
    )
    val_nme = metricas["global"]
    val_nme_olhos = metricas["olhos"]

    historico["epoca"].append(epoca + 1)
    historico["loss"].append(epoca_loss)
    for chave, valor in metricas.items():
      historico[chave].append(valor)

    print(formatar_epoca(epoca + 1, args.epochs, epoca_loss, metricas))

    if val_nme + val_nme_olhos < melhor:
      melhor = val_nme + val_nme_olhos
      melhor_epoca = epoca + 1
      melhores_metricas = metricas
      # Salvando estritamente apenas o arquivo de pesos na pasta de saída
      torch.save(model.state_dict(), out_dir / "modelo.pt")

  imprimir_resumo(melhor_epoca, melhores_metricas)
  
  # Gera os gráficos agrupados organizados na pasta de saída
  gerar_graficos_agrupados(historico, contagem_atributos, out_dir)

  with open(out_dir / "historico_treino.json", "w", encoding="utf-8") as f:
    json.dump(dict(historico), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
  main()