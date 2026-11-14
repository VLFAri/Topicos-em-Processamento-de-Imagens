import random
from pathlib import Path

import cv2
import numpy as np

TAMANHO = 256
MARGEM = 1.25

ROT_MAX = 15
ESCALA = (0.85, 1.2)
TRANS_MAX = 0.06
BRILHO = 20
CONTRASTE = (0.8, 1.2)

P_ROT = 0.5
P_ESCALA = 0.5
P_TRANS = 0.5
P_ESPELHO = 0.5
P_COR = 0.5

MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def ler_imagem_rgb(caminho):
    img = cv2.imread(str(caminho))
    if img is None:
        raise FileNotFoundError(f"Não foi possível ler a imagem: {caminho}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def carregar_mirror(data_dir):
    caminho = Path(data_dir) / "WFLW_annotations" / "Mirror98.txt"
    mirror = np.loadtxt(caminho, delimiter=",", dtype=int)
    assert mirror.shape == (98,) and (mirror[mirror] == np.arange(98)).all()
    return mirror


def matriz_recorte(caixa, margem=MARGEM):
    x0, y0, x1, y1 = caixa
    c = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    lado = margem * max(x1 - x0, y1 - y0)
    s = TAMANHO / lado
    t = TAMANHO / 2 - s * c
    return np.array([[s, 0, t[0]],
                     [0, s, t[1]]], dtype=np.float32)


def transformar_pontos(pts, M):
    return pts @ M[:, :2].T + M[:, 2]


def recortar(face, M=None):
    if M is None:
        M = matriz_recorte(face["caixa"])
    recorte = cv2.warpAffine(ler_imagem_rgb(face["imagem"]), M, (TAMANHO, TAMANHO))
    return recorte, transformar_pontos(face["landmarks"], M)


def espelhar(recorte, pts, mirror):
    recorte = recorte[:, ::-1].copy()
    pts = pts.copy()
    pts[:, 0] = TAMANHO - 1 - pts[:, 0]
    return recorte, pts[mirror]


def aumentar(recorte, pts, mirror):
    aplicados = []

    ang, f, tx, ty = 0.0, 1.0, 0.0, 0.0
    if random.random() < P_ROT:
        ang = random.uniform(-ROT_MAX, ROT_MAX)
        aplicados.append(f"rot {ang:+.0f}°")
    if random.random() < P_ESCALA:
        f = random.uniform(*ESCALA)
        aplicados.append(f"escala {f:.2f}")
    if random.random() < P_TRANS:
        tx = random.uniform(-TRANS_MAX, TRANS_MAX) * TAMANHO
        ty = random.uniform(-TRANS_MAX, TRANS_MAX) * TAMANHO
        aplicados.append("transl")

    if aplicados:
        A = cv2.getRotationMatrix2D((TAMANHO / 2, TAMANHO / 2), ang, 1 / f)
        A[:, 2] += (tx, ty)
        recorte = cv2.warpAffine(recorte, A, (TAMANHO, TAMANHO))
        pts = transformar_pontos(pts, A)

    if random.random() < P_ESPELHO:
        recorte, pts = espelhar(recorte, pts, mirror)
        aplicados.append("espelho")

    if random.random() < P_COR:
        a = random.uniform(*CONTRASTE)
        b = random.uniform(-BRILHO, BRILHO)
        recorte = np.clip(recorte.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        aplicados.append("brilho/contraste")

    return recorte, pts, aplicados


def normalizar(recorte):
    x = recorte.astype(np.float32).transpose(2, 0, 1) / 255.0
    return ((x - MEDIA) / STD).astype(np.float32)