import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from src.inference import preprocess_image_from_pil, predict_topk
from PIL import Image
import numpy as np

def test_preprocess_from_pil_shape_and_range():
    img = Image.fromarray((np.random.rand(32,32,3)*255).astype('uint8'))
    tensor = preprocess_image_from_pil(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 3, 224, 224)
    assert torch.isfinite(tensor).all()

def test_predict_topk_with_dummy_model():
    # dummy model that returns predictable logits
    class Dummy(torch.nn.Module):
        def forward(self, x):
            batch = x.shape[0]
            out = torch.zeros(batch, 10)
            out[:, 3] = 10.0  # class index 3 has highest score
            return out
    dummy = Dummy()
    x = torch.randn(1,3,224,224)
    results = predict_topk(dummy, x, device="cpu", k=3)
    assert isinstance(results, list)
    assert results[0]['class']  # name exists
    assert results[0]['prob'] > 0.9
