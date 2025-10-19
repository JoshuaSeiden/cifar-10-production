# api/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from api.model_serving import model_server

app = FastAPI(title="CIFAR-10 Image Classifier API")


# -----------------------------
# Health check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(None), top_k: int = 3):
    # Check that a file was provided
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Run inference
    try:
        image_bytes = await file.read()
        predictions = model_server.predict(image_bytes, top_k=top_k)
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")


# -----------------------------
# Classes endpoint
# -----------------------------
@app.get("/classes")
def classes():
    from src.inference import CLASS_NAMES
    return {"classes": CLASS_NAMES}
