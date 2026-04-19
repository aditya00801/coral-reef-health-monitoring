import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import gdown

# =========================
# CONFIG
# =========================
FILE_ID = "1AMHjMG81IsNlcsFB_TaV4DwA0u8BpCsq"
MODEL_PATH = "model.keras"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"

IMG_SIZE = 224
CLASS_NAMES = ["Bleached Coral", "Healthy Coral"]

# =========================
# DOWNLOAD MODEL
# =========================
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

@st.cache_resource
def load_cnn_model():
    download_model()
    return load_model(MODEL_PATH)

model = load_cnn_model()

# =========================
# PREPROCESS
# =========================
def preprocess(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# =========================
# UI
# =========================
st.title("🌊 Coral Reef Health Monitoring")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file)
    st.image(img)

    x = preprocess(img)
    pred = model.predict(x)

    label = CLASS_NAMES[np.argmax(pred)]
    confidence = np.max(pred) * 100

    st.success(f"Prediction: {label}")
    st.info(f"Confidence: {confidence:.2f}%")