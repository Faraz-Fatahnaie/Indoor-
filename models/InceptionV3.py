import torch.nn as nn
import torchvision.models as models


class InceptionV3(nn.Module):
    def __init__(self, num_classes=67):
        super(InceptionV3, self).__init__()
        self.model = models.inception_v3()
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
