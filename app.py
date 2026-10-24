import io
import json
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles

from config import FINAL_MODEL_PATH, LABEL_PATH, IMAGE_SIZE

app = FastAPI(title="Smart Waste Sorter")

# Model di-load sekali. Kalau gagal, server tetap hidup dan membalas 503.
model, class_names = None, []
try:
    model = tf.keras.models.load_model(FINAL_MODEL_PATH)
    with open(LABEL_PATH) as f:
        class_names = json.load(f)
except Exception as e:
    print("MODEL GAGAL DIMUAT:", e)


@app.get("/api/health")
def health():
    return {"model_ready": model is not None}


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(503, "Model belum tersedia")
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB").resize(IMAGE_SIZE)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Gambar tidak dapat dibaca")

    arr = np.expand_dims(np.array(image, dtype=np.float32), axis=0)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "label": class_names[idx],
        "confidence": float(probs[idx]),
        "probabilities": {n: float(p) for n, p in zip(class_names, probs)},
    }


# Harus paling bawah
app.mount("/", StaticFiles(directory="web", html=True), name="web")