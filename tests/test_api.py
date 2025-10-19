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
# Helpers
# -----------------------------
def create_dummy_image_bytes():
    """Returns a small random PNG image as bytes"""
    img = Image.fromarray((np.random.rand(32,32,3) * 255).astype("uint8"))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def post_image(file_bytes, top_k=3):
    files = {"file": ("test.png", file_bytes, "image/png")}
    return client.post(f"/predict?top_k={top_k}", files=files)


# -----------------------------
# Predict endpoint tests
# -----------------------------
def test_predict_endpoint_valid_image_default_topk():
    img_bytes = create_dummy_image_bytes()
    response = post_image(img_bytes)
    assert response.status_code == 200
    preds = response.json()["predictions"]
    assert len(preds) == 3  # default top_k
    for pred in preds:
        assert "class" in pred and "prob" in pred
        assert isinstance(pred["prob"], float)


def test_predict_endpoint_valid_image_custom_topk():
    img_bytes = create_dummy_image_bytes()
    response = post_image(img_bytes, top_k=5)
    assert response.status_code == 200
    preds = response.json()["predictions"]
    assert len(preds) == 5


def test_predict_endpoint_missing_file():
    response = client.post("/predict", files={})
    assert response.status_code == 400
    assert "detail" in response.json()


def test_predict_endpoint_invalid_file():
    files = {"file": ("test.txt", b"notanimage", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "detail" in response.json()


def test_predict_endpoint_topk_out_of_bounds():
    img_bytes = create_dummy_image_bytes()

    # top_k too small
    response = post_image(img_bytes, top_k=0)
    assert response.status_code == 400

    # top_k too large
    response = post_image(img_bytes, top_k=11)
    assert response.status_code == 400
