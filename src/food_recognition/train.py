import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from food_recognition.config import DATA_DIR, MODEL_PATH
from food_recognition.model import IMAGENET_MEAN, IMAGENET_STD, build_model, eval_transform

EPOCHS = 2
BATCH_SIZE = 32
LR = 3e-4

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                logits = model(images)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = DATA_DIR / "processed" / "food10"
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_transform)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2)

    model = build_model(len(train_ds.classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * labels.size(0)
        val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: train_loss={running_loss / len(train_ds):.4f} val_acc={val_acc:.2%}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": train_ds.classes}, MODEL_PATH)
    print(f"saved {MODEL_PATH}")


if __name__ == "__main__":  # required on Windows: DataLoader workers re-import this module
    main()
