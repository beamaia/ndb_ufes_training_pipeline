import torch.nn.functional as F
import torch.nn as nn
from torch import flatten 

from torchvision import models
from torchvision.models.densenet import densenet121
from torchvision.models.densenet import DenseNet121_Weights

class DenseNet121(nn.Module):
    def __init__(self, weights=DenseNet121_Weights.IMAGENET1K_V1, num_classes=3):
        super(DenseNet121, self).__init__()
        self.model = densenet121(weights=weights)
        self.features = self.model.features
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes) # the number of filters to learn in the first convolution layer -> number of classes 
        self.classifier = self.model.classifier

    def forward(self, x):
        out = self.model(x)
        return out