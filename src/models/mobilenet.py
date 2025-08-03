import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models import MobileNet_V2_Weights

# Embedding MobileNetv2
class MobileNetv2(nn.Module):
    def __init__(self, weights=MobileNet_V2_Weights.DEFAULT, num_classes=3):
        super(MobileNetv2, self).__init__()
        self.model = mobilenet_v2(weights=weights)
        self.features = self.model.features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.model.last_channel, num_classes),
        )
        self.classifier = self.model.classifier

    def forward(self, x):
        out = self.model(x)
        return out
