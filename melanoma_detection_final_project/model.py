# model.py
import torch.nn as nn
from torchvision import models

class MelanomaClassifier(nn.Module):
    def __init__(self):
        super(MelanomaClassifier, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)  # 2 classes: melanoma malignant and benign

    def forward(self, x):
        return self.model(x)