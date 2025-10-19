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
    data = {"top_k": str(top_k)}  # Form parameters must be strings
    return client.post("/predict", files=files, data=data)


# -----------------------------
# Predict endpoint tests
# -----------------------------
def test_predict_endpoint_valid_image_default_topk():
    """Test default top_k=3 predictions"""
    img_bytes = create_dummy_image_bytes()
    response = post_image(img_bytes)
    assert response.status_code == 200
    preds = response.json()["predictions"]
    assert len(preds) == 3
    for pred in preds:
        assert "class" in pred and "prob" in pred
        assert isinstance(pred["prob"], float)


@pytest.mark.parametrize("top_k", [1, 2, 5, 10])
def test_predict_endpoint_valid_image_custom_topk(top_k):
    """Test that top_k parameter returns correct number of predictions"""
    img_bytes = create_dummy_image_bytes()
    response = post_image(img_bytes, top_k=top_k)
    assert response.status_code == 200
    preds = response.json()["predictions"]
    assert len(preds) == top_k
    for pred in preds:
        assert "class" in pred and "prob" in pred
        assert isinstance(pred["prob"], float)


def test_predict_endpoint_missing_file():
    """No file uploaded returns 400"""
    response = client.post("/predict", files={})
    assert response.status_code == 400
    assert "detail" in response.json()


def test_predict_endpoint_invalid_file():
    """Non-image file returns 400"""
    files = {"file": ("test.txt", b"notanimage", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "detail" in response.json()


@pytest.mark.parametrize("invalid_top_k", [0, 11])
def test_predict_endpoint_topk_out_of_bounds(invalid_top_k):
    """top_k outside allowed range returns 400"""
    img_bytes = create_dummy_image_bytes()
    response = post_image(img_bytes, top_k=invalid_top_k)
    assert response.status_code == 400
    assert "detail" in response.json()
