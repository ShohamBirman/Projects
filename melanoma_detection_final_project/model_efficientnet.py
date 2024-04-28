# model_efficientnet.py
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class MelanomaClassifier(nn.Module):
    def __init__(self):
        super(MelanomaClassifier, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        num_ftrs = self.model._fc.in_features
        self.model._fc = nn.Sequential(
            nn.Dropout(p=0.5),  # Add dropout layer
            nn.Linear(num_ftrs, 2)  # 2 classes: malignant and benign
        )

    def forward(self, x):
        return self.model(x)