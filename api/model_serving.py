from pathlib import Path
import torch
from torchvision import models
import io
from PIL import Image

def load_model(weights_path: str, device: str = "cpu", num_classes: int = 10):
    device = torch.device(device)
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    state = torch.load(weights_path, map_location="cpu")
    # strip "module." if present
    new_state = {}
    for k,v in state.items():
        nk = k.replace("module.", "") if k.startswith("module.") else k
        new_state[nk] = v
    model.load_state_dict(new_state)
    model.to(device)
    model.eval()
    return model
