import torch
import torch.nn as nn
import torchvision.models as models


class LandmarkResNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Carrega a ResNet-18 pré-treinada do torchvision
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.resnet = models.resnet18(weights=weights)
        
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 98 * 2)

    def forward(self, x):
        out = self.resnet(x)
        # Redimensiona a saída para o formato (Batch_Size, 98, 2)
        return out.view(-1, 98, 2)