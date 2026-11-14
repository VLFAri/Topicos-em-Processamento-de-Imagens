import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from predict import LandmarkModel

RAIZ = Path(__file__).resolve().parent

OLHO_A = [60, 61, 63, 64, 65, 67]
OLHO_B = [72, 71, 69, 68, 75, 73]

TEMPO_PISCADA_MAX = 0.4
HISTERESE = 1.1
JANELA_PISCADAS = 60.0
PISCADAS_FADIGA = 35


def calcular_ear(olho):
  v1 = np.linalg.norm(olho[1] - olho[5])
  v2 = np.linalg.norm(olho[2] - olho[4])
  h = np.linalg.norm(olho[0] - olho[3])
  return (v1 + v2) / (2.0 * h) if h > 0 else 0.0


def ear_medio(landmarks):
  return (calcular_ear(landmarks[OLHO_A]) + calcular_ear(landmarks[OLHO_B])) / 2.0


def escolher_maior_face(faces):
  areas = np.ptp(faces[:, :, 0], axis=1) * np.ptp(faces[:, :, 1], axis=1)
  return faces[int(np.argmax(areas))]


class MonitorOlhos:

  def __init__(self, limiar_fechar: float, tempo_alerta: float):
    self.tempo_alerta = tempo_alerta
    self.limiar_fechar = limiar_fechar
    self.limiar_abrir = HISTERESE * self.limiar_fechar

    self.fechado = False
    self.t_fechou = 0.0
    self.alerta = False

    self.piscadas = 0
    self.curtos = 0
    self.prolongados = 0
    self.t_piscadas = []

  def piscadas_ultimo_minuto(self, t):
    return len([x for x in self.t_piscadas if t - x <= JANELA_PISCADAS])

  def atualizar(self, ear, t):
    if ear is not None:
      if not self.fechado and ear < self.limiar_fechar:
        self.fechado = True
        self.t_fechou = t
      elif self.fechado and ear > self.limiar_abrir:
        self._fim_do_fechamento(t)

    if self.fechado and not self.alerta and t - self.t_fechou >= self.tempo_alerta:
      self.alerta = True
      self.prolongados += 1

  def _fim_do_fechamento(self, t):
    if not self.alerta:
      if t - self.t_fechou <= TEMPO_PISCADA_MAX:
        self.piscadas += 1
        self.t_piscadas.append(t)
      else:
        self.curtos += 1
    self.fechado = False
    self.alerta = False


def put_text_with_bg(
    img, text, org, font, scale, text_color, bg_color=(0, 0, 0), thickness=2
):
  x, y = org
  (text_width, text_height), baseline = cv2.getTextSize(
      text, font, scale, thickness
  )
  top_left = (x, y - text_height - 6)
  bottom_right = (x + text_width + 6, y + baseline + 4)

  cv2.rectangle(img, top_left, bottom_right, bg_color, -1)
  cv2.putText(
      img, text, (x + 3, y), font, scale, text_color, thickness, cv2.LINE_AA
  )


def desenhar_hud(frame, monitor, ear, t, rosto_detectado):
  h, w = frame.shape[:2]
  fonte = cv2.FONT_HERSHEY_SIMPLEX

  if monitor.alerta:
    status, cor = "ALERTA: SONOLENCIA!", (0, 0, 255)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 12)
  elif monitor.piscadas_ultimo_minuto(t) > PISCADAS_FADIGA:
    status, cor = "AVISO: Fadiga Detectada (Piscadas altas)", (0, 165, 255)
  else:
    status, cor = "Normal", (0, 255, 0)

  limiar = f"{monitor.limiar_fechar:.2f}"
  olhos = "FECHADOS" if monitor.fechado else "ABERTOS"

  put_text_with_bg(frame, f"Status: {status}", (30, 50), fonte, 0.8, cor)
  put_text_with_bg(
      frame,
      f"EAR: {ear:.2f} | Limiar: {limiar} | Olhos: {olhos}",
      (30, 100),
      fonte,
      0.7,
      (255, 255, 255),
  )
  put_text_with_bg(
      frame,
      f"Piscadas: {monitor.piscadas} (ult. min: {monitor.piscadas_ultimo_minuto(t)})",
      (30, 150),
      fonte,
      0.7,
      (255, 255, 255),
  )
  put_text_with_bg(
      frame,
      f"Fechamentos curtos: {monitor.curtos} | prolongados: {monitor.prolongados}",
      (30, 200),
      fonte,
      0.7,
      (255, 255, 255),
  )

  if not rosto_detectado:
    put_text_with_bg(
        frame, "[AVISO] Rosto nao encontrado", (30, 250), fonte, 0.7, (0, 0, 255)
    )


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--source", default="0", help="indice da webcam ou arquivo de video")
  ap.add_argument("--out", default=None, help="salva o video processado")
  ap.add_argument("--no-display", action="store_true", help="nao abre janela")
  ap.add_argument("--weights", default=str(RAIZ / "pesos" / "modelo.pt"))
  ap.add_argument("--fracao", type=float, default=0.21, help="limiar de olho fechado")
  ap.add_argument(
      "--tempo-alerta",
      type=float,
      default=1.5,
      help="segundos de olhos fechados para o alerta",
  )
  args = ap.parse_args()

  webcam = args.source.isdigit()
  mostrar = not args.no_display
  device = "cuda" if torch.cuda.is_available() else "cpu"
  print(f"[INFO] Dispositivo selecionado: {device}")

  cap = cv2.VideoCapture(int(args.source) if webcam else args.source)
  if not cap.isOpened():
    print("[ERRO] Nao foi possivel abrir a fonte de video.")
    sys.exit(1)

  if webcam:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

  fps = cap.get(cv2.CAP_PROP_FPS)
  if not fps or fps != fps or fps < 1:
    fps = 30.0

  nome_janela = "Monitoramento Facial - WFLW"
  if mostrar:
    cv2.namedWindow(nome_janela, cv2.WINDOW_NORMAL)

  print(f"[INFO] Carregando o LandmarkModel com os pesos: {args.weights}")
  try:
    sistema = LandmarkModel(weights=args.weights, device=device)
  except Exception as e:
    print(f"[ERRO] Falha ao carregar o modelo: {e}")
    cap.release()
    if mostrar:
      cv2.destroyAllWindows()
    sys.exit(1)

  monitor = MonitorOlhos(args.fracao, args.tempo_alerta)
  escritor = None
  ear = 0.0
  idx = 0

  historico_t = []
  historico_ear = []
  t_inicio = time.monotonic()

  print("[INFO] Monitoramento iniciado. Pressione 'q' para sair.")

  while True:
    ok, frame = cap.read()
    if not ok:
      break

    t = time.monotonic() if webcam else idx / fps
    idx += 1

    faces = sistema.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    rosto_detectado = faces.shape[0] > 0

    ear_frame = None
    if rosto_detectado:
      landmarks = escolher_maior_face(faces)
      for x, y in landmarks:
        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
      ear_frame = ear_medio(landmarks)
      ear = ear_frame

      tempo_relativo = (time.monotonic() - t_inicio) if webcam else (idx / fps)
      historico_t.append(tempo_relativo)
      historico_ear.append(ear_frame)

    alerta_antes = monitor.alerta
    monitor.atualizar(ear_frame, t)
    if monitor.alerta and not alerta_antes:
      print("\a", end="", flush=True)

    desenhar_hud(frame, monitor, ear, t, rosto_detectado)

    if args.out:
      if escritor is None:
        altura, largura = frame.shape[:2]
        escritor = cv2.VideoWriter(
            args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (largura, altura)
        )
      escritor.write(frame)

    if mostrar:
      cv2.imshow(nome_janela, frame)
      if cv2.waitKey(1) & 0xFF == ord("q"):
        break
      if cv2.getWindowProperty(nome_janela, cv2.WND_PROP_VISIBLE) < 1:
        break

  cap.release()
  if escritor is not None:
    escritor.release()
  if mostrar:
    cv2.destroyAllWindows()

  if len(historico_ear) > 0:
    plt.figure(figsize=(10, 4))
    plt.plot(historico_t, historico_ear, color="tab:blue", label="EAR Previsto")
    plt.axhline(
        y=monitor.limiar_fechar,
        color="red",
        linestyle="--",
        label=f"Limiar de Fechamento ({monitor.limiar_fechar:.2f})",
    )
    plt.xlabel("Tempo (s)")
    plt.ylabel("EAR (Eye Aspect Ratio)")
    plt.title("Evolução Temporal do EAR na Execução do Vídeo")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    caminho_grafico = RAIZ / "grafico_ear_video.png"
    plt.savefig(caminho_grafico, dpi=200)
    plt.close()
    print(f"[INFO] Gráfico temporal do EAR salvo em: {caminho_grafico}")

  print(
      f"[INFO] Resumo: {monitor.piscadas} piscadas | {monitor.curtos} fechamentos"
      f" curtos | {monitor.prolongados} prolongados"
  )


if __name__ == "__main__":
  main()