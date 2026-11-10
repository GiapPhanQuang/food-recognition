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


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # new head for our classes
    return model


class Predictor:
    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        checkpoint = torch.load(model_path, map_location=device)
        self.classes: list[str] = checkpoint["classes"]
        self.device = device
        self.model = build_model(len(self.classes), pretrained=False)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(device).eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        x = eval_transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = self.model(x).softmax(dim=1)[0]
        conf, idx = probs.max(dim=0)
        return {"label": self.classes[idx.item()], "confidence": round(conf.item(), 4)}
