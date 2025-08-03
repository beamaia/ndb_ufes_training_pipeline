import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16_bn
from torchvision.models.vgg import VGG16_BN_Weights

class VGG16(nn.Module):
    def __init__(self, weights=VGG16_BN_Weights.IMAGENET1K_V1, num_classes=3, dropout=0.5):
        super(VGG16, self).__init__()
        self.model = vgg16_bn(weights=weights)
        self.features = self.model.features
        self.model.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        self.classifier = self.model.classifier

    def forward(self, x):
        out = self.model(x)
        return out
