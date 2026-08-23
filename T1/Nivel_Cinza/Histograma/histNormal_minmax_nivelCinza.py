import numpy as np
from PIL import Image
import os
from pathlib import Path
 
dir_script = Path(__file__).resolve().parent
pasta_imagens = dir_script.parent.parent / "Vistex"

def calcula_minmax(idx):
    img = Image.open(pasta_imagens / pngs[idx]).convert("L")
    histograma = img.histogram()
    histograma = np.array(histograma)

    return ((histograma - histograma.min()) / (histograma.max() - histograma.min())) * 255

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