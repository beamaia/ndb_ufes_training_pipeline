import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from torchvision.models.resnet import ResNet50_Weights

# Embedding ResNet50 (From: https://github.com/avilash/pytorch-siamese-triplet)
class ResNet50(nn.Module):
    def __init__(self, weights=ResNet50_Weights.IMAGENET1K_V1, num_classes=3):
        super(ResNet50, self).__init__()

        self.model = resnet50(weights=weights)
        self.features = nn.Sequential(
            self.model.conv1,
            self.model.bn1,
            self.model.relu,
            self.model.maxpool,
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
            self.model.avgpool
        )

        self.model.fc = nn.Linear(512 * 4, num_classes)
        self.classifier = self.model.fc
        
        # # Fix blocks
        # for p in self.features[0].parameters():
        #     p.requires_grad = False
        # for p in self.features[1].parameters():
        #     p.requires_grad = False

        # def set_bn_fix(m):
        #     classname = m.__class__.__name__
        #     if classname.find('BatchNorm') != -1:
        #         for p in m.parameters(): p.requires_grad = False

        # self.features.apply(set_bn_fix)

    def forward(self, x):
        out = self.model(x)
        return out