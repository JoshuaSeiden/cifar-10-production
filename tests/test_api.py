import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import io
import pytest
from fastapi.testclient import TestClient
from api.main import app
from PIL import Image
import numpy as np

client = TestClient(app)


# -----------------------------
# Health check
# -----------------------------
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# -----------------------------
# Predict endpoint
# -----------------------------
def create_dummy_image_bytes():
    """Returns a small random PNG image as bytes"""
    img = Image.fromarray((np.random.rand(32,32,3) * 255).astype("uint8"))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_predict_endpoint_valid_image():
    img_bytes = create_dummy_image_bytes()
    files = {"file": ("test.png", img_bytes, "image/png")}
    response = client.post("/predict", files=files)

    assert response.status_code == 200
    json_resp = response.json()
    assert "predictions" in json_resp
    assert isinstance(json_resp["predictions"], list)
    assert len(json_resp["predictions"]) > 0
    # Each prediction should have 'class' and 'prob'
    for pred in json_resp["predictions"]:
        assert "class" in pred and "prob" in pred
        assert isinstance(pred["prob"], float)


def test_predict_endpoint_missing_file():
    response = client.post("/predict", files={})
    assert response.status_code == 400
    assert "detail" in response.json()


def test_predict_endpoint_invalid_file():
    files = {"file": ("test.txt", b"notanimage", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "detail" in response.json()
