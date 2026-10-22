# Script para fazer treinamento, validação e teste

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from download import marcador
from datasets import gerar_datasets
from arquiteturas.MLP_CNN import MLP_CNN_Modelo
from arquiteturas.MLP_CNN_com_BatchNorm_Dropout import MLP_CNN_com_BatchNorm_Dropout_Modelo
from arquiteturas.ResNet import ResNet_Modelo

def treino_validacao():
    if "download concluído!" in marcador:

        lotes = int(input("Número de batches: "))

        taxa_aprendizado = float(input("Taxa de aprendizado: "))

        otim = 0
        while True:
            print("Escolha um otimizador -> ", end="")
            opt = int(input("Escolha um número de 0 e 4 [0-SGD, 1-Adam, 2-AdamW, 3-RAdam,4-NAdam]: "))
            if 0 <= opt <= 4:
                otim = opt
                break
            else:
                print("Não há esse número na lista de otimizadores, escolha entre 0 e 4")

        loader_treino, loader_validacao, loader_teste = gerar_datasets(lotes)
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Rodando em {device}')
        modelos_disponiveis = [MLP_CNN_Modelo, MLP_CNN_com_BatchNorm_Dropout_Modelo, ResNet_Modelo]
        epocas = int(input("Quantidade de épocas que deseja treinar: "))
        
        for mod in modelos_disponiveis:
            print("")
            print("")
            print(f'Nova arquitetura/modelo: {mod.__name__}')
            print("")
            modelo = mod().to(device)
            ce = nn.CrossEntropyLoss()

            otimizador = None
            match otim:
                case 0:
                    otimizador = optim.SGD(modelo.parameters(), lr=taxa_aprendizado)
                case 1:
                    otimizador = optim.Adam(modelo.parameters(), lr=taxa_aprendizado)
                case 2:
                    otimizador = optim.AdamW(modelo.parameters(), lr=taxa_aprendizado)
                case 3:
                    otimizador = optim.RAdam(modelo.parameters(), lr=taxa_aprendizado)
                case 4:
                    otimizador = optim.NAdam(modelo.parameters(), lr=taxa_aprendizado)

            historico = {"loss_treino": [], "acuracia_treino": [], "loss_validacao": [], "acuracia_validacao": []}

            for epoca in range(epocas):
                print("")
                modelo.train()
                soma_loss, acertos, total = 0.0, 0, 0

                for inputs, labels in loader_treino:
                    inputs, labels = inputs.to(device), labels.to(device)
        
                    otimizador.zero_grad()
                    outputs = modelo(inputs)
                    loss = ce(outputs, labels)
        
                    loss.backward()
                    otimizador.step()
        
                    soma_loss += loss.item() * len(labels)
                    acertos += (outputs.argmax(1) == labels).sum().item()
                    total += len(labels)

                acuracia_treino = acertos / total
                loss_treino = soma_loss / total

                modelo.eval()
                soma_loss, acertos, total = 0.0, 0, 0
                with torch.no_grad():
                    for inputs, labels in loader_validacao:
                        inputs, labels = inputs.to(device), labels.to(device)
            
                        outputs = modelo(inputs)
                        soma_loss += ce(outputs, labels).item() * len(outputs)
            
                        acertos += (outputs.argmax(1) == labels).sum().item()
                        total += len(labels)
                    loss_validacao = soma_loss / total
                    acuracia_validacao = acertos / total

                historico["loss_treino"].append(loss_treino)
                historico["acuracia_treino"].append(acuracia_treino)
                historico["loss_validacao"].append(loss_validacao)
                historico["acuracia_validacao"].append(acuracia_validacao)

                print(f"Época {epoca+1} de {epocas}:")
                print(f"Loss do treino: {loss_treino:.8f}")
                print(f"Acurácia do treino: {acuracia_treino * 100:.2f}%")
                print(f"Loss de validação: {loss_validacao:.8f}")
                print(f"Acurácia de validação: {acuracia_validacao * 100:.2f}%")

            # Avaliação final no conjunto de teste para o modelo atual
            modelo.eval()
            soma_loss, acertos, total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in loader_teste:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = modelo(inputs)
                    soma_loss += ce(outputs, labels).item() * len(outputs)
                    acertos += (outputs.argmax(1) == labels).sum().item()
                    total += len(labels)
                
            loss_teste = soma_loss / total
            acuracia_teste = acertos / total

            print("")
            print("=" * 50)
            print(f"Resultado no Teste para {mod.__name__}:")
            print(f"Loss do teste: {loss_teste:.8f}")
            print(f"Acurácia do teste: {acuracia_teste * 100:.2f}%")
            print("=" * 50)

            # Gráfico do modelo atual (gerado após todas as épocas)
            eixo_epocas = range(1, epocas + 1)
            nome_modelo = mod.__name__
            fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))
            fig.suptitle(f"{nome_modelo}")

            ax_loss.plot(eixo_epocas, historico["loss_treino"], marker="o", label="Treino")
            ax_loss.plot(eixo_epocas, historico["loss_validacao"], marker="o", label="Validação")
            ax_loss.set_title("Loss")
            ax_loss.set_xlabel("Época")
            ax_loss.set_ylabel("Loss")
            ax_loss.grid(True, alpha=0.3)
            ax_loss.legend()

            ax_acc.plot(eixo_epocas, [a * 100 for a in historico["acuracia_treino"]], marker="o", label="Treino")
            ax_acc.plot(eixo_epocas, [a * 100 for a in historico["acuracia_validacao"]], marker="o", label="Validação")
            ax_acc.set_title("Acurácia")
            ax_acc.set_xlabel("Época")
            ax_acc.set_ylabel("Acurácia (%)")
            ax_acc.grid(True, alpha=0.3)
            ax_acc.legend()

            fig.tight_layout()
            fig.savefig(f"grafico_{nome_modelo}.png", dpi=150)

        plt.show()  

    else:
        print("Download não concluído ou não realizado!")

if __name__ == "__main__":
    treino_validacao()