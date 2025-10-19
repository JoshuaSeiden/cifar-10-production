import io
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F

CLASS_NAMES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# Preprocessing pipeline
_preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def preprocess_image_from_pil(image: Image.Image, device="cpu"):
    """
    Convert a PIL image to a normalized tensor ready for model inference.
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image, got {type(image)}")
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = _preprocess(image).unsqueeze(0).to(device)
    return tensor


def preprocess_image_from_bytes(b: bytes, device="cpu"):
    """
    Convert image bytes to tensor ready for model inference.
    """
    try:
        img = Image.open(io.BytesIO(b))
    except Exception as e:
        raise ValueError(f"Could not read image bytes: {e}")
    return preprocess_image_from_pil(img, device=device)


def predict_topk(model: torch.nn.Module, input_tensor: torch.Tensor, device="cpu", k=3):
    """
    Predict top-k classes for a single image tensor.
    Returns a list of dictionaries with 'class' and 'prob'.
    """
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(input_tensor)}")
    if input_tensor.dim() != 4 or input_tensor.size(0) != 1 or input_tensor.size(1) != 3:
        raise ValueError(f"Expected input tensor of shape (1,3,H,W), got {input_tensor.shape}")
    if k < 1 or k > len(CLASS_NAMES):
        raise ValueError(f"k must be between 1 and {len(CLASS_NAMES)}")

    model.to(device)
    input_tensor = input_tensor.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        topk = torch.topk(probs, k)

    return [{"class": CLASS_NAMES[i], "prob": float(topk.values[0][j])}
            for j, i in enumerate(topk.indices[0])]
