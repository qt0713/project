import torch
import torch.nn as nn
from torchvision import models

try:
    import timm
except ImportError:
    timm = None


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int, crop_size: int = 224):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * (crop_size // 8) * (crop_size // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class ResNetInceptionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.branch3 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=3, padding=1)
        self.branch5 = nn.Conv2d(in_channels, out_channels // 4, kernel_size=5, padding=2)
        self.branch_pool = nn.Conv2d(in_channels, out_channels // 4, kernel_size=1)
        self.pool = nn.MaxPool2d(3, stride=1, padding=1)
        self.conv_res = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        bp = self.branch_pool(self.pool(x))
        out = torch.cat([b1, b3, b5, bp], dim=1)
        out = self.bn(self.conv_res(out))
        if out.shape == x.shape:
            out = out + x
        return self.relu(out)


class ResNetInception(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.block1 = ResNetInceptionBlock(32, 64)
        self.block2 = ResNetInceptionBlock(64, 128)
        self.block3 = ResNetInceptionBlock(128, 128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def get_model(model_name: str, num_classes: int, crop_size: int = 224) -> nn.Module:
    if model_name == "simple":
        return SimpleCNN(num_classes=num_classes, crop_size=crop_size)
    if model_name == "vgg16":
        model = models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    if model_name == "vgg19":
        model = models.vgg19(pretrained=False)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    if model_name == "resnet18":
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "resnet50":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "inception":
        model = models.inception_v3(pretrained=False, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "mobilenet":
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    if model_name == "resnet_inception":
        return ResNetInception(num_classes)
    if model_name == "inception_resnet_v2":
        if timm is None:
            raise ImportError("Please install timm: pip install timm")
        model = timm.create_model("inception_resnet_v2", pretrained=False)
        if hasattr(model, "classif"):
            model.classif = nn.Linear(model.classif.in_features, num_classes)
        elif hasattr(model, "fc"):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            raise ValueError("Unknown classifier head for inception_resnet_v2")
        return model
    raise ValueError(f"Unknown model: {model_name}")
