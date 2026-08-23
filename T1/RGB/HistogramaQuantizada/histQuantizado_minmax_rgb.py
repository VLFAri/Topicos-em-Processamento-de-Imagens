import numpy as np
from PIL import Image
import os
from pathlib import Path
 
dir_script = Path(__file__).resolve().parent
pasta_imagens = dir_script.parent.parent / "Vistex"

def quantizar(hist):
    histograma_quantizado = np.array([])

    for c in range(0,256,2):
       histograma_quantizado = np.append(histograma_quantizado, hist[c:c+2].sum())
    
    return histograma_quantizado

def calcula_minmax(idx):
    img = Image.open(pasta_imagens / pngs[idx])
    histograma = img.histogram()
    histograma = np.array(histograma)

    histograma_vermelho_quantizado = quantizar(histograma[0:256])
    histograma_verde_quantizado = quantizar(histograma[256:512])
    histograma_azul_quantizado = quantizar(histograma[512:768])

    minmax_vermelho = ((histograma_vermelho_quantizado - histograma_vermelho_quantizado.min()) / (histograma_vermelho_quantizado.max() - histograma_vermelho_quantizado.min())) * 255
    minmax_verde = ((histograma_verde_quantizado - histograma_verde_quantizado.min()) / (histograma_verde_quantizado.max() - histograma_verde_quantizado.min())) * 255
    minmax_azul = ((histograma_azul_quantizado - histograma_azul_quantizado.min()) / (histograma_azul_quantizado.max() - histograma_azul_quantizado.min())) * 255

    return np.concatenate([minmax_vermelho, minmax_verde, minmax_azul])

if __name__ == "__main__":
    pngs = sorted(os.listdir(pasta_imagens))
    pngs_array = np.array(pngs)    
    n = int(input("Entrada N definida no trabalho (escolha de preferência um número ímpar): "))
    matriz = np.zeros((864, 864))

    qualidade_do_programa = 0
    print("")
    for i in range(864):
        minmax_entrada = calcula_minmax(i)

        for j in range(i+1,864,1):
            minmax_busca = calcula_minmax(j)

            distancia = np.sqrt(np.sum((minmax_entrada - minmax_busca) ** 2))

            matriz[i][j] = distancia
            matriz[j][i] = distancia


        imagens_ordenadas = np.argsort(matriz[i])

        imagens_ordenadas = imagens_ordenadas[imagens_ordenadas != i]

        top_n = imagens_ordenadas[:n]

        nomes_top_n = pngs_array[top_n] 
        distancias_top_n = matriz[i][top_n]
        print(f"============ {pngs[i]} ============")
        print("Imagens mais parecidas e suas distãncias euclidianas:")
        acerto = 0
        erro = 0
        for prints in range(n):
            print(f"{prints+1}- {nomes_top_n[prints]} -> Distância: {distancias_top_n[prints]:.3f}")
            if pngs[i][:4] == nomes_top_n[prints][:4]:
                acerto +=1
            else:
                erro += 1
        if acerto > erro:
            qualidade_do_programa += 1
            print(f"A imagem {pngs[i]} apresentou acerto nesse script")
        else:
            print(f"A imagem {pngs[i]} apresentou erro nesse script")
        print("")

    porcentagem = (qualidade_do_programa * 1e2) / 864
    print(f"O programa acertou a classe de {qualidade_do_programa} imagens e obteve {porcentagem:.1f}% de acerto")