"""
A arquitetura adotada baseia-se na ResNet e é composta por 3
 camadas convolucionais seguidas por quatro camadas fully-connected
 (MLP). A função de ativação ReLU é utilizada para introduzir 
 não linearidade ao modelo. Além disso, são empregadas as técni-
cas de Batch Normalization e Dropout, com taxa de 40%, para redu-
zir o sobreajuste (overfitting) e favorecer a capacidade de gene-
ralização da rede neural. A arquitetura ResNet também utiliza 
conexões residuais, que facilitam o treinamento de redes profundas.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ResNet_Modelo(nn.Module):
    def __init__(self):
        super(ResNet_Modelo, self).__init__()
        self.c1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.b1 = nn.BatchNorm2d(64)
        self.p1 = nn.MaxPool2d(2, 2)
        self.c2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.b2 = nn.BatchNorm2d(128)
        self.p2 = nn.MaxPool2d(2, 2)
        self.c3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.b3 = nn.BatchNorm2d(256)
        self.p3 = nn.MaxPool2d(2, 2)

        # Camadas novas para o bloco residual (2ª conv + BatchNorm + atalho 1x1 de cada estágio)
        self.c1b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.c2b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.c3b = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.b1b = nn.BatchNorm2d(64)
        self.b2b = nn.BatchNorm2d(128)
        self.b3b = nn.BatchNorm2d(256)

        self.s1 = nn.Conv2d(3, 64, kernel_size=1)
        self.s2 = nn.Conv2d(64, 128, kernel_size=1)
        self.s3 = nn.Conv2d(128, 256, kernel_size=1)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32,10)

        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        identidade = self.s1(x)
        x = F.relu(self.b1(self.c1(x)))
        x = self.b1b(self.c1b(x))
        x = F.relu(x + identidade)
        x = self.p1(x)

        identidade = self.s2(x)
        x = F.relu(self.b2(self.c2(x)))
        x = self.b2b(self.c2b(x))
        x = F.relu(x + identidade)
        x = self.p2(x)

        identidade = self.s3(x)
        x = F.relu(self.b3(self.c3(x)))
        x = self.b3b(self.c3b(x))
        x = F.relu(x + identidade)
        x = self.p3(x)

        #Achata (B, 256, 4, 4) -> (B, 4096) antes das camadas lineares
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        return x