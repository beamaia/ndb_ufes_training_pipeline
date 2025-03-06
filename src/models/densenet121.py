import torch.nn.functional as F
import torch.nn as nn
from torch import flatten 

from torchvision import models
from torchvision.models.densenet import densenet121

class DenseNet121(nn.Module):
    def __init__(self, weights=models.DenseNet121_Weights.IMAGENET1K_V1, num_classes=3):
        super(DenseNet121, self).__init__()
        self.model = densenet121(weights=weights)
        self.features = self.model.features
        self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes) # the number of filters to learn in the first convolution layer -> number of classes
    
    def forward(self, x):
        # features = self.features(x)
        
        # out = F.relu(features, inplace=True)
        # out = F.adaptive_avg_pool2d(out, (1, 1))
        # out = flatten(out, 1)
        # out = self.classifier(out)
        out = self.model(x)

        return out