#Script para treino e validação do aprendizagem da rede neural (usada para todas as arquiteturas e estratégias de melhoria)
 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MLP_CNN_com_BatchNorm_Dropout_Modelo(nn.Module):
    #O método __init__ é o construtor da classe, onde são definidas todas as camadas da rede.
    def __init__(self):
        super(MLP_CNN_com_BatchNorm_Dropout_Modelo, self).__init__()
        
        self.c1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.b1 = nn.BatchNorm2d(64)
        self.p1 = nn.MaxPool2d(2, 2)
        self.c2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.b2 = nn.BatchNorm2d(128)
        self.p2 = nn.MaxPool2d(2, 2)
        self.c3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.b3 = nn.BatchNorm2d(256)
        self.p3 = nn.MaxPool2d(2, 2)
        

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32,10)

        self.dropout = nn.Dropout(0.4)
 
    def forward(self, x):

        x = F.relu(self.b1(self.c1(x)))
        x = self.p1(x)
        x = F.relu(self.b2(self.c2(x)))
        x = self.p2(x)
        x = F.relu(self.b3(self.c3(x)))
        x = self.p3(x)
 
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
 