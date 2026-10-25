"""
A arquitetura adotada é composta por 3 camadas convolucionais,
 seguidas por 4 camadas *fully connected* (MLP). A função 
 de ativação ReLU (*Rectified Linear Unit*) é utilizada para 
 introduzir não linearidade ao modelo. Não foram empregadas 
 técnicas de regularização ou mecanismos específicos de aprimora-
 mento da arquitetura, como Dropout, Batch Normalization ou
  conexões residuais, presentes em arquiteturas como a ResNet.
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MLP_CNN_Modelo(nn.Module):
    def __init__(self):
        super(MLP_CNN_Modelo, self).__init__()
        
        self.c1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.p1 = nn.MaxPool2d(2, 2)
        self.c2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.p2 = nn.MaxPool2d(2, 2)
        self.c3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.p3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32,10)

 
    def forward(self, x):

        x = F.relu(self.c1(x))
        x = self.p1(x)
        x = F.relu(self.c2(x))
        x = self.p2(x)
        x = F.relu(self.c3(x))
        x = self.p3(x)
 
        #Achata (flatten) os dados: transforma o tensor 4D (batch, canais, altura, largura) em 2D (batch, total_de_valores_por_imagem)
        x = torch.flatten(x, 1)
 
        x = self.fc1(x)
        x = F.relu(x)
 
        x = self.fc2(x)
        x = F.relu(x)
 
        x = self.fc3(x)
        x = F.relu(x)

        x = self.fc4(x)
        return x 
 