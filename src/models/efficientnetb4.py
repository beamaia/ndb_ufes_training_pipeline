import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.efficientnet import efficientnet_b4
from torchvision.models.efficientnet import EfficientNet_B4_Weights

class EfficientNetB4(nn.Module):
    def __init__(self, weights=EfficientNet_B4_Weights, num_classes=3):
        super(EfficientNetB4, self).__init__()
        self.model = efficientnet_b4(weights=EfficientNet_B4_Weights.IMAGENET1K_V1)
        self.features = self.model.features
        
        lastconv_output_channels = self.model.classifier[1].in_features  
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(lastconv_output_channels, num_classes),
        )
        self.classifier = self.model.classifier

    def forward(self, x):
        out = self.model(x)
        return x
