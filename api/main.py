# from fastapi import FastAPI, File, UploadFile, Query
# from fastapi.responses import JSONResponse
# import os
# from api.model_serving import load_model
# from src.inference import preprocess_image_from_bytes, predict_topk

# app = FastAPI(title="CIFAR10 API")

# MODEL_PATH = os.environ.get("MODEL_PATH", "models/cifar10_resnet50.pth")
# DEVICE = os.environ.get("DEVICE", "cpu")

# @app.on_event("startup")
# def load():
#     global model
#     # If model file missing, create a minimal placeholder model to keep tests fast
#     if not os.path.exists(MODEL_PATH):
#         from src.model import get_resnet50_model
#         model = get_resnet50_model(num_classes=10)
#     else:
#         model = load_model(MODEL_PATH, device=DEVICE)

# @app.get("/health")
# def health():
#     return {"status": "ok"}

# @app.post("/predict")
# async def predict(file: UploadFile = File(...), top_k: int = Query(3, ge=1, le=10)):
#     try:
#         contents = await file.read()
#         tensor = preprocess_image_from_bytes(contents)
#         preds = predict_topk(model, tensor, device=DEVICE, k=top_k)
#         return JSONResponse({"predictions": preds})
#     except Exception as e:
#         return JSONResponse({"error": str(e)}, status_code=500)
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import os
from api.model_serving import load_model
from src.inference import preprocess_image_from_bytes, predict_topk

MODEL_PATH = os.environ.get("MODEL_PATH", "models/cifar10_resnet50.pth")
DEVICE = os.environ.get("DEVICE", "cpu")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model at startup and clean up on shutdown."""
    global model
    print("Loading model...")

    if not os.path.exists(MODEL_PATH):
        from src.model import get_resnet50_model
        model = get_resnet50_model(num_classes=10)
    else:
        model = load_model(MODEL_PATH, device=DEVICE)

    print("Model loaded successfully.")
    yield
    print("App shutdown complete.")

app = FastAPI(title="CIFAR10 API", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = Query(3, ge=1, le=10)):
    try:
        contents = await file.read()
        tensor = preprocess_image_from_bytes(contents)
        preds = predict_topk(model, tensor, device=DEVICE, k=top_k)
        return JSONResponse({"predictions": preds})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
