from pathlib import Path

import cv2
import numpy as np
import torch

from auxiliares.preprocessamento import TAMANHO, matriz_recorte, normalizar
from auxiliares.resnet18 import LandmarkResNet

RAIZ = Path(__file__).resolve().parent


class LandmarkModel:
  """Modelo de regressão direta para detecção de landmarks faciais (98 pontos)."""

  def __init__(self, weights: str, device: str = "cpu"):
    self.device = torch.device(device)

    caminho = Path(weights)
    if not caminho.exists():
      caminho = RAIZ / weights

    # pretrained=False evita download na internet durante avaliação offline (Seção 11.2)
    self.landmark_model = LandmarkResNet(pretrained=False)
    self.landmark_model.load_state_dict(
        torch.load(caminho, map_location=self.device, weights_only=True)
    )
    self.landmark_model.to(self.device)
    self.landmark_model.eval()

    self.detector = cv2.FaceDetectorYN.create(
        model=str(RAIZ / "modelos" / "face_detection_yunet_2023mar.onnx"),
        config="",
        input_size=(320, 320),
        score_threshold=0.5,
        nms_threshold=0.3,
    )

  def predict(self, image: np.ndarray) -> np.ndarray:
    imagem = np.ascontiguousarray(image)
    h_orig, w_orig = imagem.shape[:2]
    self.detector.setInputSize((w_orig, h_orig))

    _, faces = self.detector.detect(cv2.cvtColor(imagem, cv2.COLOR_RGB2BGR))

    if faces is None or len(faces) == 0:
      return np.empty((0, 98, 2), dtype=np.float32)

    matrizes, recortes = [], []

    for face in faces:
      x, y, w, h = face[:4]
      if w <= 0 or h <= 0:
        continue

      caixa = np.array([x, y, x + w, y + h], dtype=np.float32)
      M = matriz_recorte(caixa)
      recorte = cv2.warpAffine(imagem, M, (TAMANHO, TAMANHO))

      matrizes.append(M)
      recortes.append(normalizar(recorte))

    if not recortes:
      return np.empty((0, 98, 2), dtype=np.float32)

    lote = torch.from_numpy(np.stack(recortes)).to(self.device)

    with torch.no_grad():
      pred = self.landmark_model(lote).cpu().numpy() * TAMANHO

    resultado = []
    for pts, M in zip(pred, matrizes):
      M_inv = cv2.invertAffineTransform(M)
      resultado.append(pts @ M_inv[:, :2].T + M_inv[:, 2])

    return np.array(resultado, dtype=np.float32)