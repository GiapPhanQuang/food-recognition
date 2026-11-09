"""Download Food-101 and copy a 10-class subset into ImageFolder layout."""

import json
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
]
TRAIN_PER_CLASS = 300
VAL_PER_CLASS = 100
OUT_DIR = DATA_DIR / "processed" / "food10"


def copy_split(food_dir: Path, meta_file: str, split: str, per_class: int) -> None:
    meta = json.loads((food_dir / "meta" / meta_file).read_text())
    for cls in CLASSES:
        out = OUT_DIR / split / cls
        out.mkdir(parents=True, exist_ok=True)
        for item in meta[cls][:per_class]:  # item looks like "pho/1234567"
            src = food_dir / "images" / f"{item}.jpg"
            shutil.copy2(src, out / src.name)
        print(f"{split}/{cls}: {per_class} images")


def main() -> None:
    raw_dir = DATA_DIR / "raw"
    Food101(root=raw_dir, download=True)  # ~5 GB download
    food_dir = raw_dir / "food-101"
    copy_split(food_dir, "train.json", "train", TRAIN_PER_CLASS)
    copy_split(food_dir, "test.json", "val", VAL_PER_CLASS)


if __name__ == "__main__":
    main()
