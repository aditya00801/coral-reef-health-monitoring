from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import gdown

app = Flask(__name__)

# ======================
# MODEL CONFIG
# ======================
FILE_ID = "1AMHjMG81IsNlcsFB_TaV4DwA0u8BpCsq"
MODEL_PATH = "model.keras"

IMG_SIZE = 224
CLASS_NAMES = ["Bleached Coral", "Healthy Coral"]

# ======================
# DOWNLOAD MODEL
# ======================
def download_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)

download_model()
model = load_model(MODEL_PATH)

# ======================
# PREPROCESS
# ======================
def preprocess(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# ======================
# ROUTES
# ======================
@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        file = request.files["file"]
        img = Image.open(file)

        x = preprocess(img)
        pred = model.predict(x)

        label = CLASS_NAMES[np.argmax(pred)]
        confidence = float(np.max(pred) * 100)

        result = f"{label} ({confidence:.2f}%)"

    return render_template("index.html", result=result)

# ======================
# RUN
# ======================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)