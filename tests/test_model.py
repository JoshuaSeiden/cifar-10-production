import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from src.model import get_resnet18_model

def test_model_instantiation():
    model = get_resnet18_model(num_classes=10)
    assert isinstance(model, nn.Module)
    assert model.fc.out_features == 10

def test_all_layers_trainable():
    model = get_resnet18_model(num_classes=10)
    req = [p.requires_grad for p in model.parameters()]
    assert all(req), "Expected all parameters to be trainable (True)"
