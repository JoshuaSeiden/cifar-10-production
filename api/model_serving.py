import torch
from pathlib import Path
from src.model import get_resnet18_model
from src.inference import preprocess_image_from_bytes, predict_topk


class ModelServer:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_resnet18_model(num_classes=10)

        # Path to local weights inside your project
        model_path = Path("models/cifar_10_resnet18.pth")

        if not model_path.exists():
            raise FileNotFoundError(f"Model weights not found at {model_path}")

        print(f"Loading local model weights from {model_path}")
        state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_bytes: bytes, top_k: int = 3):
        tensor = preprocess_image_from_bytes(image_bytes, device=self.device)
        return predict_topk(self.model, tensor, device=self.device, k=top_k)


# Singleton instance for FastAPI
model_server = ModelServer()
