from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from food_recognition.api import app, get_predictor


class FakePredictor:
    def predict(self, image: Image.Image) -> dict:
        return {"label": "pho", "confidence": 0.99}


@pytest.fixture
def client():
    app.dependency_overrides[get_predictor] = FakePredictor
    yield TestClient(app)
    app.dependency_overrides.clear()


def make_jpeg() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (64, 64), color="red").save(buf, format="JPEG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_label(client):
    r = client.post("/predict", files={"file": ("test.jpg", make_jpeg(), "image/jpeg")})
    assert r.status_code == 200
    assert r.json() == {"label": "pho", "confidence": 0.99}


def test_predict_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("x.txt", b"not an image", "text/plain")})
    assert r.status_code == 400
    assert r.json() == {"detail": "File is not a valid image"}
