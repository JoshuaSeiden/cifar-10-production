import io
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F

CLASS_NAMES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

_preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def preprocess_image_from_pil(image: Image.Image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    t = _preprocess(image).unsqueeze(0)
    return t

def preprocess_image_from_bytes(b: bytes):
    img = Image.open(io.BytesIO(b)).convert("RGB")
    return preprocess_image_from_pil(img)

def predict_topk(model, input_tensor, device="cpu", k=3):
    model.to(device)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        topk = torch.topk(probs, k)
        vals = topk.values.cpu().numpy().tolist()[0]
        idxs = topk.indices.cpu().numpy().tolist()[0]
    return [{"class": CLASS_NAMES[i], "prob": float(vals[j])} for j,i in enumerate(idxs)]
