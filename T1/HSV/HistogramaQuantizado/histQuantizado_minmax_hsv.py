import numpy as np
from PIL import Image, ImageOps
import os
from pathlib import Path
 
dir_script = Path(__file__).resolve().parent
pasta_imagens = dir_script.parent.parent / "Vistex"

def quantizar(hist):
    histograma_quantizado = np.array([])

    for c in range(0,256,2):
       histograma_quantizado = np.append(histograma_quantizado, hist[c:c+2].sum())
    
    return histograma_quantizado
    
def calcula_pdf(idx):
    img = Image.open(pasta_imagens / pngs[idx]).convert('HSV')
    h,s,v = img.split()
    v_eq = ImageOps.equalize(v)
    hist_h = np.array(h.histogram())       
    hist_s = np.array(s.histogram())    
    hist_v = np.array(v_eq.histogram())

    hist_h_quantizado = quantizar(hist_h)
    hist_s_quantizado = quantizar(hist_s)
    hist_v_quantizado = quantizar(hist_v)   

    minmax_h = ((hist_h_quantizado - hist_h_quantizado.min()) / (hist_h_quantizado.max() - hist_h_quantizado.min())) * 255
    minmax_s = ((hist_s_quantizado - hist_s_quantizado.min()) / (hist_s_quantizado.max() - hist_s_quantizado.min())) * 255
    minmax_v = ((hist_v_quantizado - hist_v_quantizado.min()) / (hist_v_quantizado.max() - hist_v_quantizado.min())) * 255

    return np.concatenate([minmax_h, minmax_s, minmax_v])

if __name__ == "__main__":
    pngs = sorted(os.listdir(pasta_imagens))
    pngs_array = np.array(pngs)    
    n = int(input("Entrada N definida no trabalho (escolha de preferência um número ímpar): "))
    matriz = np.zeros((864, 864))

    qualidade_do_programa = 0
    print("")

    for i in range(864): 

        pdf_entrada = calcula_pdf(i)

        for j in range(i+1,864,1):
            pdf_busca = calcula_pdf(j)

            distancia = np.sqrt(np.sum((pdf_entrada - pdf_busca) ** 2))

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