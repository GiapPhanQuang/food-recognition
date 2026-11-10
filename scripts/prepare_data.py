"""Copy a 10-class Food-101 subset into ImageFolder layout: train / val / test."""

import json
import random
import shutil
from pathlib import Path

from torchvision.datasets import Food101

from food_recognition.config import DATA_DIR

CLASSES = [
    "pho",
    "spring_rolls",
    "fried_rice",
    "dumplings",
    "ramen",
    "sushi",
    "pizza",
    "hamburger",
    "steak",
    "ice_cream",
]  # keep your existing list
VAL_PER_CLASS = 150  # carved out of Food-101's train list
SEED = 42
OUT_DIR = DATA_DIR / "processed" / "food10"


def copy_images(food_dir: Path, items: list[str], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for item in items:  # item looks like "pho/1234567"
        src = food_dir / "images" / f"{item}.jpg"
        shutil.copy2(src, dest / src.name)


def main() -> None:
    raw_dir = DATA_DIR / "raw"
    food_dir = raw_dir / "food-101"
    if not food_dir.exists():
        Food101(root=raw_dir, download=True)  # ~5 GB download

    train_meta = json.loads((food_dir / "meta" / "train.json").read_text())
    test_meta = json.loads((food_dir / "meta" / "test.json").read_text())
    rng = random.Random(SEED)  # fixed seed -> same split every time

    for cls in CLASSES:
        items = list(train_meta[cls])
        rng.shuffle(items)
        copy_images(food_dir, items[VAL_PER_CLASS:], OUT_DIR / "train" / cls)
        copy_images(food_dir, items[:VAL_PER_CLASS], OUT_DIR / "val" / cls)
        copy_images(food_dir, test_meta[cls], OUT_DIR / "test" / cls)
        n_train, n_test = len(items) - VAL_PER_CLASS, len(test_meta[cls])
        print(f"{cls}: train={n_train} val={VAL_PER_CLASS} test={n_test}")


if __name__ == "__main__":
    main()
