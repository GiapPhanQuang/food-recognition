import argparse
import time

import mlflow
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from food_recognition.config import DATA_DIR, MLFLOW_TRACKING_URI, MODELS_DIR
from food_recognition.model import ARCHS, IMAGENET_MEAN, IMAGENET_STD, build_model, eval_transform

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune an ImageNet model on food10")
    p.add_argument("--arch", default="resnet18", choices=list(ARCHS))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--warmup-epochs", type=int, default=1)  # must be >= 1
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device) -> float:
    model.train()
    loss_sum, n = 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # per batch, not per epoch
        loss_sum += loss.item() * labels.size(0)
        n += labels.size(0)
    return loss_sum / n


def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")  # plain CE, no smoothing, for val
    loss_sum, correct, total = 0.0, 0, 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                logits = model(images)
            loss_sum += criterion(logits.float(), labels).item()
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    data_dir = DATA_DIR / "processed" / "food10"
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_transform)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)

    model = build_model(args.arch, len(train_ds.classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    steps = len(train_loader)
    total_steps = args.epochs * steps
    warmup_steps = max(1, args.warmup_epochs * steps)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps),
            CosineAnnealingLR(optimizer, T_max=max(1, total_steps - warmup_steps)),
        ],
        milestones=[warmup_steps],
    )
    scaler = torch.amp.GradScaler(device, enabled=device == "cuda")

    ckpt_path = MODELS_DIR / f"{args.arch}_food10.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("food10")
    with mlflow.start_run(run_name=f"{args.arch}-e{args.epochs}-lr{args.lr}"):
        mlflow.log_params(vars(args))
        mlflow.log_params(
            {
                "device": device,
                "torch_version": torch.__version__,
                "num_train": len(train_ds),
                "num_params": sum(p.numel() for p in model.parameters()),
            }
        )
        best_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler, scaler, device
            )
            val_loss, val_acc = evaluate(model, val_loader, device)
            elapsed = time.perf_counter() - start
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "lr": scheduler.get_last_lr()[0],
                    "epoch_time_s": elapsed,
                },
                step=epoch,
            )
            print(
                f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.2%} ({elapsed:.0f}s)"
            )
            if val_acc > best_acc:  # keep the best epoch, not the last one
                best_acc = val_acc
                torch.save(
                    {
                        "arch": args.arch,
                        "state_dict": model.state_dict(),
                        "classes": train_ds.classes,
                    },
                    ckpt_path,
                )

        mlflow.log_metric("best_val_acc", best_acc)
        if device == "cuda":
            mlflow.log_metric("peak_vram_mb", torch.cuda.max_memory_allocated() / 2**20)
        mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoint")
        print(f"best val_acc={best_acc:.2%}, saved {ckpt_path}")


if __name__ == "__main__":  # required on Windows: DataLoader workers re-import this module
    main()
