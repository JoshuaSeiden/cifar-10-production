import torch
from src.model import get_resnet18_model
from src.inference import preprocess_image_from_bytes, predict_topk
from pathlib import Path

class ModelServer:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_resnet18_model(num_classes=10)

        cache_dir = Path("/tmp/torch/hub/checkpoints")
        cache_dir.mkdir(parents=True, exist_ok=True)

        model_path = cache_dir / "cifar_10_resnet18.pth"

        if not model_path.exists():
            print(f"Downloading model to {model_path}…")
            url = "https://huggingface.co/jseiden/cifar10-resnet18/resolve/main/cifar_10_resnet18.pth" # get fine tuned weights from huggingface
            state_dict = torch.hub.load_state_dict_from_url(url, map_location=self.device)
            torch.save(state_dict, model_path)
        else:
            print(f"Loading model from cache {model_path}…")
            state_dict = torch.load(model_path, map_location=self.device)
        
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_bytes: bytes, top_k: int = 3):
        tensor = preprocess_image_from_bytes(image_bytes, device=self.device)
        return predict_topk(self.model, tensor, device=self.device, k=top_k)


# Singleton instance for the API
model_server = ModelServer()
