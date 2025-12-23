from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# -----------------------
# Flask app
# -----------------------
app = Flask(__name__)

# -----------------------
# Load model ONCE (IMPORTANT)
# -----------------------
MODEL_PATH = "crop_disease_model.h5"

model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------
# Class labels (CHANGE according to your training)
# -----------------------
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

# -----------------------
# Home route
# -----------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# -----------------------
# Predict route
# -----------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return "No file uploaded"

        file = request.files["file"]

        # Image preprocessing
        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))

        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)
        class_index = int(np.argmax(prediction))
        result = class_labels[class_index]

        return render_template("index.html", prediction=result)

    except Exception as e:
        return f"Error occurred: {e}"

# -----------------------
# DO NOT use app.run()
# Render uses gunicorn
# -----------------------
