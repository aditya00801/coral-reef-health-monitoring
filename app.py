from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
import requests
import os

app = Flask(__name__)

MODEL_PATH = "model.h5"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1AMHjMG81IsNlcsFB_TaV4DwA0u8BpCsq"

# ✅ Download model if not exists
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("Model downloaded!")

# ✅ Load model once
model = None

def get_model():
    global model
    if model is None:
        download_model()
        model = tf.keras.models.load_model(MODEL_PATH)
    return model

@app.route('/')
def home():
    return "Coral Reef API is running 🚀"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        image = Image.open(file.stream).resize((224, 224))

        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        model = get_model()
        prediction = model.predict(img_array)

        result = int(prediction[0][0] > 0.5)

        return jsonify({
            "prediction": result
        })

    except Exception as e:
        return jsonify({"error": str(e)})