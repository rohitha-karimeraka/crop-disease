from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Model path
MODEL_PATH = "crop_disease_model.h5"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Class names
class_names = [
    "Bacterial leaf blight",
    "Blight",
    "Brown spot",
    "Common Rust",
    "Gray Leaf Spot",
    "Leaf smut",
    "Bacterial Blight",
    "Curl Virus",
    "Fusarium Wilt",
    "Healthy"
]

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = int(np.argmax(prediction))
    result = class_names[class_index]

    return render_template("index.html", label=result)

# ❌ app.run() use cheyyoddu (Render lo)
