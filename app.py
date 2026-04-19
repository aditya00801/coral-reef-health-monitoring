from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
import os

app = Flask(__name__)

# -------------------------
# MODEL CONFIG
# -------------------------
MODEL_PATH = "model.h5"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1AMHjMG81IsNlcsFB_TaV4DwA0u8BpCsq"

model = None

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        with requests.get(MODEL_URL, stream=True) as r:
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
        print("Model downloaded!")

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

        img = np.array(image) / 255.0
        img = np.expand_dims(img, axis=0)

        model = get_model()
        p = float(model.predict(img)[0][0])  # probability of class-1

        # 👉 Map classes:
        # assume p = probability of "Healthy"
        healthy = p * 100
        bleached = (1 - p) * 100

        return jsonify({
            "healthy": round(healthy, 2),
            "bleached": round(bleached, 2),
            "label": "Healthy Coral" if healthy > bleached else "Bleached Coral"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500