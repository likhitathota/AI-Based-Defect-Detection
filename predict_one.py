import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# ---------- 1. Settings ----------
IMG_SIZE = (128, 128)
MODEL_PATH = "tile_defect_classifier.h5"

# NOTE:
# When we loaded the data for training, Keras sorted folders alphabetically:
# ['defect', 'good']  -> label 0 = defect, 1 = good
class_names = ["defect", "good"]


def load_and_prepare_image(img_path):
    """Load image from path, resize and prepare as model input."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0   # scale to 0–1
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array


def main():
    # ---------- 2. Choose an image to test ----------
    # Try with a good image first:
    img_path = "dataset/test/good/good_0.png"
    # To test a defect image, change to for example:
    # img_path = "dataset/test/defect/defect_0.png"

    print(f"Using image: {img_path}")

    # ---------- 3. Load model and make prediction ----------
    model = load_model(MODEL_PATH)

    img_array = load_and_prepare_image(img_path)
    prob = model.predict(img_array)[0][0]  # single value between 0 and 1

    # threshold 0.5: >0.5 => class 1 (good), <=0.5 => class 0 (defect)
    pred_label = 1 if prob > 0.5 else 0
    pred_class_name = class_names[pred_label]

    print(f"Raw probability (for 'good'): {prob:.3f}")
    print(f"Predicted class: {pred_class_name}")


if __name__ == "__main__":
    main()
