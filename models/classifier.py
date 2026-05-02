import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, vgg16, resnet50


# ---------------------------------------------------------------------------
# ConvNeXt-Tiny  (Proposed / Primary model)
# ---------------------------------------------------------------------------
class ConvNeXtClassifier(nn.Module):
    def __init__(self, num_classes=4, input_channels=6):
        super().__init__()
        self.backbone = convnext_tiny(pretrained=True)
        # Replace first conv to accept 6-channel input (original + masked)
        self.backbone.features[0][0] = nn.Conv2d(
            input_channels, 96, kernel_size=4, stride=4
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(768, eps=1e-6),
            nn.Linear(768, num_classes),
        )

    def forward(self, x):
        x = self.backbone.features(x)
        x = x.mean([-2, -1])   # Global average pooling
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# VGG-16 baseline
# ---------------------------------------------------------------------------
class VGG16Classifier(nn.Module):
    def __init__(self, num_classes=4, input_channels=6):
        super().__init__()
        base = vgg16(pretrained=True)
        orig_conv = base.features[0]
        new_conv = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = orig_conv.weight
            new_conv.weight[:, 3:, :, :] = orig_conv.weight
        base.features[0] = new_conv
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# ResNet-50 baseline
# ---------------------------------------------------------------------------
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=4, input_channels=6):
        super().__init__()
        base = resnet50(pretrained=True)
        orig_conv = base.conv1
        new_conv = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = orig_conv.weight
            new_conv.weight[:, 3:, :, :] = orig_conv.weight
        base.conv1 = new_conv
        in_feats = base.fc.in_features
        base.fc = nn.Linear(in_feats, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# MLP / BetterMLP  (used by Firefly pipeline)
# ---------------------------------------------------------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class BetterMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.layers(x)
