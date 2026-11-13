import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/food_recognition/config.py -> repo root
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODELS_DIR / "resnet18_food10.pt"))

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)
