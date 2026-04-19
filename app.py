from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
import os

app = Flask(__name__)

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "model.h5"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1AMHjMG81IsNlcsFB_TaV4DwA0u8BpCsq"

model = None
history = []

# -------------------------
# DOWNLOAD MODEL
# -------------------------
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        with requests.get(MODEL_URL, stream=True) as r:
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print("Model downloaded!")

# -------------------------
# LOAD MODEL (LAZY)
# -------------------------
def get_model():
    global model
    if model is None:
        download_model()
        print("Loading model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded!")
    return model

# -------------------------
# ROUTES
# -------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        image = Image.open(file.stream).resize((224, 224))

        # preprocess
        img = np.array(image) / 255.0
        img = np.expand_dims(img, axis=0)

        model = get_model()
        p = float(model.predict(img)[0][0])  # probability

        # mapping
        healthy = p * 100
        bleached = (1 - p) * 100

        label = "Healthy Coral" if healthy > bleached else "Bleached Coral"

        result = {
            "healthy": round(healthy, 2),
            "bleached": round(bleached, 2),
            "label": label
        }

        # store history
        history.insert(0, result)

        return jsonify(result)

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


@app.route('/history')
def get_history():
    return jsonify(history[:10])


# -------------------------
# RUN (for local only)
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)