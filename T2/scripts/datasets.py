"""Script para separar as imagens em datasets diferentes: treino, validação e teste

AVISO: ESSE SCRIPT SÓ EXECUTARÁ CORRETAMENTE SE O SCRIPT baixar_dados.py JÁ ESTIVER SIDO EXECUTADO E OS ARQUIVOS .mat BAIXADOS """
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.io import loadmat
import numpy as np
from pathlib import Path

# Mesma pasta usada por baixar_dados.py (onde ficam os arquivos .mat)
RAIZ = Path("data")

def converter(caminho_dados):
    dados = loadmat(caminho_dados)
    X = torch.from_numpy(np.transpose(dados["X"], (3, 2, 0, 1)).astype(np.float32) / 255.0)
    y = torch.from_numpy(dados["y"].flatten().astype(np.int64) % 10)  # rótulo 10 vira 0
    return TensorDataset(X, y)

def gerar_datasets(batch):
    # Carrega os dados de treino e divide em subconjuntos de treino e validação
    dados_treino = converter(RAIZ / "train_32x32.mat")
    
    tamanho_val = int(len(dados_treino) * 0.2)
    tamanho_treino = len(dados_treino) - tamanho_val
    
    treino, validacao = random_split(dados_treino, [tamanho_treino, tamanho_val])
    
    teste = converter(RAIZ / "test_32x32.mat")
    
    dataset_treino = DataLoader(treino, batch_size=batch, shuffle=True, num_workers=2)
    dataset_validacao = DataLoader(validacao, batch_size=batch, shuffle=False, num_workers=2)
    dataset_teste = DataLoader(teste, batch_size=batch, shuffle=False, num_workers=2)

    return dataset_treino, dataset_validacao, dataset_teste