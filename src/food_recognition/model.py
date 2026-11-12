from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Shared by training-eval and serving, so preprocessing can never drift apart.
eval_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


ARCHS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT),
}


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    factory, weights = ARCHS[arch]
    model = factory(weights=weights if pretrained else None)
    if hasattr(model, "fc"):  # ResNet: head is model.fc
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:  # ConvNeXt / MobileNetV3: head is the last layer of model.classifier
        last = model.classifier[-1]
        model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    return model


class Predictor:
    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        checkpoint = torch.load(model_path, map_location=device)
        self.classes: list[str] = checkpoint["classes"]
        self.device = device
        arch = checkpoint.get("arch", "resnet18")  # v0.1.0 checkpoints have no "arch" key
        self.model = build_model(arch, len(self.classes), pretrained=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(device).eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        x = eval_transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = self.model(x).softmax(dim=1)[0]
        conf, idx = probs.max(dim=0)
        return {"label": self.classes[idx.item()], "confidence": round(conf.item(), 4)}
