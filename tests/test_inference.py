import io
import pytest
import torch
from PIL import Image
from src.inference import (
    preprocess_image_from_pil,
    preprocess_image_from_bytes,
    predict_topk,
    CLASS_NAMES
)


# -----------------------------
# Helper functions
# -----------------------------
def get_dummy_image(size=(32,32)):
    """Returns a random RGB PIL image"""
    import numpy as np
    return Image.fromarray((np.random.rand(*size,3)*255).astype('uint8'))


def get_dummy_bytes(size=(32,32)):
    img = get_dummy_image(size)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def get_dummy_model():
    import torch.nn as nn
    class DummyModel(nn.Module):
        def forward(self, x):
            out = torch.zeros(1, len(CLASS_NAMES))
            out[0, 3] = 10.0  # class index 3 is highest
            return out
    return DummyModel()


# -----------------------------
# Preprocessing tests
# -----------------------------
def test_preprocess_pil_valid():
    img = get_dummy_image()
    tensor = preprocess_image_from_pil(img)
    assert tensor.shape == (1, 3, 224, 224)
    assert torch.isfinite(tensor).all()


def test_preprocess_pil_invalid_type():
    with pytest.raises(TypeError):
        preprocess_image_from_pil("not an image")


def test_preprocess_bytes_valid():
    b = get_dummy_bytes()
    tensor = preprocess_image_from_bytes(b)
    assert tensor.shape == (1, 3, 224, 224)


def test_preprocess_bytes_invalid():
    with pytest.raises(ValueError):
        preprocess_image_from_bytes(b"notanimage")


# -----------------------------
# Prediction tests
# -----------------------------
def test_predict_topk_basic():
    model = get_dummy_model()
    x = torch.randn(1, 3, 224, 224)
    results = predict_topk(model, x, k=3)
    assert isinstance(results, list)
    assert all("class" in r and "prob" in r for r in results)
    assert results[0]["class"] in CLASS_NAMES
    assert results[0]["prob"] > 0.9


def test_predict_topk_invalid_tensor():
    model = get_dummy_model()
    x = torch.randn(3, 224, 224)  # missing channel
    with pytest.raises(ValueError):
        predict_topk(model, x)


def test_predict_topk_invalid_k():
    model = get_dummy_model()
    x = torch.randn(1, 3, 224, 224)
    with pytest.raises(ValueError):
        predict_topk(model, x, k=20)
