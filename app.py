from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

MODEL_PATH = "crop_disease_model.h5"
model = None   # 👈 IMPORTANT

class_names = [
    "Bacterial leaf blight",
    "Blight",
    "Brown spot",
    "Common_Rust",
    "Gray_Leaf_Spot",
    "Leaf smut",
    "bacterial_blight",
    "curl_virus",
    "fussarium_wilt",
    "healthy"
]

def load_model_once():
    global model
    if model is None:
        print("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    load_model_once()   # 👈 model first request lo load avvutundi

    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = int(np.argmax(prediction))
    result = class_names[class_index]

    return render_template("index.html", label=result)
