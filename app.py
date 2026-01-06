from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# ----------------------------
# Model path
# ----------------------------
MODEL_PATH = "crop_disease_model.h5"
model = None   # model only once load avvadaniki

# ----------------------------
# Class names (training order lo undali)
# ----------------------------
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

# ----------------------------
# Thresholds (IMPORTANT)
# ----------------------------
CONFIDENCE_THRESHOLD = 0.75
DIFFERENCE_THRESHOLD = 0.30

# ----------------------------
# Load model only once
# ----------------------------
def load_model_once():
    global model
    if model is None:
        print("🔄 Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully")

# ----------------------------
# Home page
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html")

# ----------------------------
# Prediction route
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    load_model_once()

    if "image" not in request.files:
        return render_template("index.html", label="❌ No file uploaded")

    file = request.files["image"]

    if file.filename == "":
        return render_template("index.html", label="❌ No file selected")

    try:
        # Image processing
        img = Image.open(file).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        prediction = model.predict(img_array)[0]

        # Top probabilities
        sorted_probs = np.sort(prediction)[::-1]
        top1 = float(sorted_probs[0])
        top2 = float(sorted_probs[1])

        class_index = int(np.argmax(prediction))

        # Decision logic
        if top1 < CONFIDENCE_THRESHOLD or (top1 - top2) < DIFFERENCE_THRESHOLD:
            result = "❌ Sorry, we can’t predict this image"
        else:
            result = f" Disease: {class_names[class_index]}"

        return render_template("index.html", label=result)

    except Exception as e:
        print("Error:", e)
        return render_template("index.html", label="❌ Error processing image")

# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
