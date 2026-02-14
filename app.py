import streamlit as st
import numpy as np
import cv2
import os
import gdown
from tensorflow.keras.models import load_model

MODEL_PATH = "coral_model.h5"
FILE_ID = "1Cj8MPPBpAenbkKR_pk2adjWlLc3C2UIZ"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

model = load_cnn_model()

st.title("Coral Reef Health Monitoring System")

uploaded_file = st.file_uploader("Upload Coral Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.resize(image, (224,224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)
    class_index = np.argmax(prediction)

    result = "Bleached Coral" if class_index == 0 else "Healthy Coral"

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.success(f"Prediction: {result}")
