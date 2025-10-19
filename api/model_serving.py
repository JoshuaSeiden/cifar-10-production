import torch
from src.model import get_resnet50_model
from src.inference import preprocess_image_from_bytes, predict_topk
from pathlib import Path

MODEL_PATH = Path("models/cifar_10_resnet50.pth")

class ModelServer:
    def __init__(self, model_path=MODEL_PATH, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_resnet50_model(num_classes=10)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_bytes: bytes, top_k: int = 3):
        tensor = preprocess_image_from_bytes(image_bytes, device=self.device)
        return predict_topk(self.model, tensor, device=self.device, k=top_k)


# Singleton instance for the API
model_server = ModelServer()
