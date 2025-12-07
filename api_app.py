from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

# ---------- 1. App + model setup ----------
app = FastAPI(title="Tile Defect Detection API")

MODEL_PATH = "tile_defect_classifier.h5"
IMG_SIZE = (128, 128)

# label 0 = defect, 1 = good  (same as in predict_one.py)
class_names = ["defect", "good"]

# load model once when API starts
model = load_model(MODEL_PATH)


def prepare_image(image_bytes: bytes):
    """Convert uploaded bytes to preprocessed image tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# ---------- 2. Health check endpoint ----------
@app.get("/")
def root():
    return {"message": "Tile  detection API is running"}


# ---------- 3. Prediction endpoint ----------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # read file bytes
    image_bytes = await file.read()

    # preprocess
    img_array = prepare_image(image_bytes)

    # predict
    prob = float(model.predict(img_array)[0][0])   # probability of "good"
    pred_label = 1 if prob > 0.5 else 0
    pred_class = class_names[pred_label]

    return JSONResponse({
        "probability_good": prob,
        "predicted_class": pred_class
    })
