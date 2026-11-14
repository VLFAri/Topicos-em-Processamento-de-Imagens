# Trabalho 3 (Final) de Tópicos em Processamento de Imagens

Facial Landmark Detection (WFLW, 98 pontos) para detecção de sinais de sonolência em vídeo.

## Estrutura dos arquivos

```text
ari/
├── README.md
├── relatorio.pdf
├── video_demo.mp4
├── train.py
├── predict.py
├── app.py
├── figuras.py
├── avaliar_teste.py
├── heatmap.py
├── requirements.txt
├── requirements-cpu.txt
├── graficos_teste/
├── auxiliares/
│   ├── dataset.py
│   ├── preprocessamento.py
│   └── resnet18.py
├── pesos/
│   └── modelo.pt
└── modelos/
    └── face_detection_yunet_2023mar.onnx
```

Os arquivos `figuras.py`, `hedatmap.py` e `avaliar_teste.py` são auxiliares e servem para gerar as figuras do relatório.

Todos os comandos abaixo devem ser executados a partir da pasta `ari/`.

## Preparação do ambiente

**OBS:** Todos os comandos foram executados no ZorinOS (distro Linux baseada no Ubuntu). É necessário ter Python 3.

Neste trabalho, foi utilizado o ambiente virtual `venv`, disponibilizado pelo Python 3, para gerenciar as dependências de forma isolada:

```bash
# Instalação do módulo venv
sudo apt install python3-venv

# Criação do ambiente virtual
python3 -m venv .venv

# Ativação do ambiente virtual
source .venv/bin/activate
```

## Instalação das dependências

Dentro do ambiente virtual `.venv`, instale as dependências com o `requirements.txt` ou com o `requirements-cpu.txt`:

```bash
# Execute esse comando SOMENTE SE houver GPU na máquina (CUDA 12.1)
pip install -r requirements.txt

# Caso não tenha GPU, a máquina executa pela CPU
pip install -r requirements-cpu.txt
```

**OBS:** O `requirements-cpu.txt` também funciona em máquinas com GPU, mas o treinamento do modelo de landmarks será executado na CPU e ficará muito mais lento. 

## Modelo de detecção de faces (YuNet)

O detector de faces **YuNet** é usado pelo `predict.py` (e, portanto, pelo `app.py`) apenas para obter a caixa da face. O arquivo (cerca de 230 KB) deve ficar na pasta `modelos/`:

```bash
curl -fL -o modelos/face_detection_yunet_2023mar.onnx https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

O nome do arquivo precisa ser exatamente `face_detection_yunet_2023mar.onnx`. 

## Dataset WFLW

O treino espera uma pasta com a estrutura original do WFLW. Neste trabalho, o dataset está em `src/data/`:

```text
src/data/
├── WFLW_images/
└── WFLW_annotations/
    ├── list_98pt_rect_attr_train_test/
    │   ├── list_98pt_rect_attr_train.txt
    │   └── list_98pt_rect_attr_test.txt
    └── Mirror98.txt
```


## Treinamento

```bash
python3 ari/train.py --data caminho/para/WFLW --out pasta_de_saida
```

Exemplo usado neste trabalho, salvando os pesos direto na pasta `pesos/`:

```bash
python3 ari/train.py --data src/data/ --out pesos/
```

- `--data`: pasta que contém `WFLW_images/` e `WFLW_annotations/` (veja a estrutura acima).
- `--out`: pasta onde o treino salva o melhor modelo (`modelo.pt`). Com `--out pesos/`, os pesos ficam em `pesos/modelo.pt`, que é o caminho que o `app.py` usa por padrão. O arquivo é atualizado sempre que a validação melhora.

O treino usa apenas o arquivo `list_98pt_rect_attr_train.txt`. O conjunto de teste do WFLW não é usado no treino nem na escolha da melhor época: a validação é uma parte do treino, separada por imagem. A cada época, o terminal mostra a perda de treino e o NME de validação (global e dos olhos).

**NOTA:**: Ao término do treinamento, o script salva os pesos da rede neural no arquivo `modelo.pt`, armazenando a versão que apresentou o melhor desempenho na validação, considerando a soma do NME global e do NME dos olhos. Adicionalmente, os gráficos de evolução da \textit{loss} de treino, do NME global, da NME por região facial e da NME por subconjunto de dificuldade são salvos automaticamente na pasta de saída (dentro de uma subpasta \texttt{graficos/}), garantindo a compatibilidade mesmo em ambientes sem interface gráfica. Esses pesos e arquivos gerados serão posteriormente utilizados pelo sistema de monitoramento e análise de desempenho.

O backbone é inicializado com os pesos da ImageNet, que o `torchvision` baixa na primeira execução. Por isso, retreinar exige internet ou esses pesos já em cache. O `predict.py` e o `app.py` não acessam a internet.

### Configuração do modelo final

| Item | Valor |
|---|---|
| Semente aleatória | 0 (`random`, `numpy` e `torch`) |
| Arquitetura | ResNet-18 (ImageNet), com a camada final substituída por `Linear(512, 196)` |
| Representação | regressão direta das 98 coordenadas (x, y), divididas por 256 |
| Backbone | fine-tuning (nenhuma camada congelada) |
| Entrada | recorte quadrado de 256 x 256, RGB, normalização da ImageNet |
| Recorte | quadrado com o centro da caixa e lado igual a 1,25 vezes o maior lado da caixa |
| Origem da caixa | retângulo anotado do WFLW (treino) e caixa do YuNet (vídeo) |
| Validação | 10% das imagens do treino, dividido por imagem |
| Otimizador | AdamW, taxa de aprendizado 0.0001 |
| Batch / épocas | 64 / 100 |
| Função de perda | MSE nas coordenadas normalizadas, com peso 5 nos landmarks 60 a 75 (olhos) |
| Melhor época | menor soma do NME global com o NME dos olhos na validação |
| Data augmentation | espelhamento (reordenado com o `Mirror98.txt`), rotação até 15°, escala de 0,85 a 1,2, translação até 6%, brilho e contraste; cada um com probabilidade 0,5, sorteado de forma independente |

Esses valores estão fixos em `train.py` e em `auxiliares/preprocessamento.py`.

**Nota:** o `train.py`, o `predict.py` e o `app.py` usam funções do diretório `auxiliares/`. Mantenha essa pasta junto com eles.

## Uso do modelo (`predict.py`)

```python
import cv2
from predict import LandmarkModel

modelo = LandmarkModel("pesos/modelo.pt", device="cpu")
imagem = cv2.cvtColor(cv2.imread("foto.jpg"), cv2.COLOR_BGR2RGB)
landmarks = modelo.predict(imagem)
```

A imagem entra em RGB, `uint8`, com formato (altura, largura, 3). Com `N = 0`, nenhuma face foi encontrada.

## Aplicação em vídeo (sonolência)

Com os pesos em `pesos/modelo.pt` e o YuNet em `modelos/`, execute o `app.py`:

```bash
# webcam
python3 ari/app.py --source 0

# arquivo de vídeo, salvando o resultado e sem abrir janela
python3 ari/app.py --source video.mp4 --out saida.mp4 --no-display
```

| Argumento | Descrição | Padrão |
|---|---|---|
| `--source` | índice da webcam ou caminho de um arquivo de vídeo | `0` |
| `--out` | salva o vídeo processado (landmarks, EAR, estado e alerta) | desativado |
| `--no-display` | processa sem abrir janela | desativado |
| `--weights` | pesos do modelo | `pesos/modelo.pt` |
| `--tempo-alerta` | segundos de olhos fechados para disparar o alerta | 1,5 |

O `app.py` também aceita `--calib` e `--fracao`, mas, como o limiar do EAR é fixo (veja abaixo), eles não têm efeito.

Quando uma face é detectada, o sistema desenha os 98 `landmarks` sobre a imagem e mostra o EAR, o limiar, o estado dos olhos (abertos ou fechados), os contadores de piscadas e fechamentos e o alerta, quando ativo (borda vermelha e texto).

Como o sistema decide:

1. **EAR:** média dos dois olhos, com os pontos (60, 61, 63, 64, 65, 67) e (72, 71, 69, 68, 75, 73).
2. **Limiar fixo:** o olho é considerado fechado quando o EAR fica abaixo de **0,16**. Para considerá-lo aberto de novo, o EAR precisa passar de 0,176 (limiar 10% maior, a histerese).
3. **Análise temporal:** o tempo vem do relógio do sistema (webcam) ou do índice do frame e do fps (arquivo de vídeo). Cada fechamento é classificado como:
   - **piscada:** até 0,4 s;
   - **fechamento curto:** mais de 0,4 s e abaixo do tempo de alerta;
   - **fechamento prolongado:** a partir de `--tempo-alerta` (1,5 s por padrão), o que gera o alerta.
4. **Sem face detectada:** o sistema mantém o último estado e avisa na tela.

### Escolha do limiar

O limiar de 0,21 foi escolhido a partir do gráfico do EAR em um vídeo de teste (`grafico_ear_video.png`): com os olhos abertos, o EAR previsto ficou em torno de 0,27 a 0,28, e nos fechamentos, em torno de 0,20 a 0,21. Como o EAR varia de pessoa para pessoa e com a posição da câmera, o valor pode precisar de ajuste para outro usuário.

### Encerramento e gráfico do EAR

Para encerrar, pressione a tecla `q` ou feche a janela. Ao terminar a execução (tecla `q`, janela fechada ou fim do vídeo), o sistema imprime no terminal o resumo das piscadas e dos fechamentos detectados e salva o gráfico `grafico_ear_video.png`, na mesma pasta do `app.py`. Esse gráfico mostra o EAR ao longo do tempo e o limiar de fechamento. Nos trechos em que nenhuma face foi detectada, o gráfico liga os pontos vizinhos com uma reta. Encerrar o processo à força (por exemplo, pelo terminal) impede a geração do gráfico.

## Geração de imagens (relatório)

Os arquivos `figuras.py`, `heatmap.py` e `avaliar_teste.oy` geram as imagens usadas no relatório. Ambos são executados a partir da pasta `ari/`.

### `figuras.py`

O modo `pre` gera **n figuras**, com *n* inteiro definido pelo usuário, de 1 a 7500 (o treino tem 7500 faces). Cada figura mostra três etapas do pré-processamento de uma face do treino: a imagem original do dataset (com a caixa anotada e o quadrado do recorte), a face recortada e o recorte após o aumento de dados.

O aumento é sorteado de forma aleatória e pode envolver rotação, mudança de escala, translação, espelhamento horizontal e alteração de brilho e contraste. Os pontos 60 a 67 aparecem em vermelho e os 68 a 75 em azul.

```bash
python3 ari/figuras.py pre --data src/data --out figuras/imagens_aumentadas --n 3
```

As figuras são salvas na pasta indicada em `--out`. Também é possível usar `--seed` (sorteio das faces) e `--mostrar` (abre as figuras na tela).

### `heatmap.py`

Gera *heatmaps* (mapas de calor) **alvo**, calculados a partir dos landmarks anotados: cada ponto vira uma gaussiana em um mapa 64x64. As figuras servem para ilustrar essa representação e a resolução na região dos olhos. O modelo treinado usa regressão direta e não prevê heatmaps.

Cada figura mostra o recorte com os landmarks, o heatmap de um landmark (o 60, por padrão), o máximo dos 98 mapas sobre o recorte e o olho direito ampliado com a grade do heatmap.

```bash
# <n> deve ser um inteiro de 1 a 2500 (o teste tem 2500 faces)
python3 ari/heatmap.py --data src/data --out figuras/heatmap --n <n>
```

As imagens são salvas na pasta indicada em `--out`. Também é possível usar `--sigma`, `--mapa`, `--landmark`, `--seed` e `--mostrar`.

*NOTA*: as faces usadas no heatmap são as faces de teste, apenas para ilustração. O teste não é usado no treino nem na escolha de hiperparâmetros.

### `avaliar_teste.py `

O arquivo avaliar_teste.py executa a avaliação rigorosa do modelo no conjunto de teste oficial do WFLW, calculando métricas como NME Global, NME dos olhos, taxas de falha, desempenho por regiões e subconjuntos de dificuldade, além da correlação do EAR. Em seguida, ele gera e salva automaticamente gráficos e figuras visuais (incluindo acertos e falhas) para compor o relatório do projeto. As imagens são salvas em `graficos_teste/`

```bash
python3 ari/avaliar_teste.py --data src/data --out graficos_teste/
```

## Verificação antes da entrega

A partir da raiz do projeto da disciplina:

```bash
python3 ari/verificar_entrega.py caminho/para/ari
```

A entrega só está no formato correto quando o verificador termina com `Entrega OK`.