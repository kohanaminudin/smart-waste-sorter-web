import argparse
import json
import numpy as np
import tensorflow as tf
from PIL import Image

from config import FINAL_MODEL_PATH, LABEL_PATH, IMAGE_SIZE


def predict(model, class_names, path):
    image = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    arr = np.expand_dims(np.array(image, dtype=np.float32), axis=0)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))

    print(f"\n{path}")
    print(f"Predicted : {class_names[idx]} ({probs[idx]:.2%})")
    for name, p in zip(class_names, probs):
        print(f"  {name}: {p:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="path gambar")
    args = parser.parse_args()

    model = tf.keras.models.load_model(FINAL_MODEL_PATH)
    with open(LABEL_PATH) as f:
        class_names = json.load(f)

    for img in args.images:
        predict(model, class_names, img)