from functools import lru_cache
from io import BytesIO
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from food_recognition.config import MODEL_PATH
from food_recognition.model import Predictor

app = FastAPI(title="Food Recognition", version="0.1.0")


@lru_cache
def get_predictor() -> Predictor:
    return Predictor(MODEL_PATH)  # loaded once, on first request


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict(
    file: Annotated[UploadFile, File()],
    predictor: Annotated[Predictor, Depends(get_predictor)],
) -> dict:
    try:
        image = Image.open(BytesIO(file.file.read()))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File is not a valid image") from None
    return predictor.predict(image)
