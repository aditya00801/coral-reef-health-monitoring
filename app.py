import streamlit as st
import numpy as np
import cv2
import os
import gdown
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

# ----------------------------
# MODEL DOWNLOAD SECTION
# ----------------------------

MODEL_PATH = "coral_model.h5"
FILE_ID = "1Cj8MPPBpAenbkKR_pk2adjWlLc3C2UIZ"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_PATH)

model = load_cnn_model()

# ----------------------------
# PAGE TITLE
# ----------------------------

st.title("🌊 Coral Reef Health Monitoring System")

# ----------------------------
# TABS
# ----------------------------

tab1, tab2 = st.tabs(["🔍 Prediction", "📊 Model Performance"])

# ----------------------------
# TAB 1 — PREDICTION
# ----------------------------

with tab1:

    uploaded_file = st.file_uploader("Upload Coral Image", type=["jpg","png","jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.resize(image, (224,224))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        start_time = time.time()
        prediction = model.predict(image, verbose=0)
        end_time = time.time()

        inference_time = round(end_time - start_time, 4)

        class_index = np.argmax(prediction)
        result = "Bleached Coral" if class_index == 0 else "Healthy Coral"

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.success(f"Prediction: {result}")
        st.info(f"Inference Time: {inference_time} seconds")

# ----------------------------
# TAB 2 — MODEL PERFORMANCE
# ----------------------------

with tab2:

    st.subheader("Model Evaluation Metrics")

    # Replace these with your real values
    accuracy = 0.92
    precision = 0.91
    recall = 0.93
    f1_score = 0.92

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", f"{accuracy*100:.2f}%")
    col2.metric("Precision", f"{precision*100:.2f}%")
    col3.metric("Recall", f"{recall*100:.2f}%")
    col4.metric("F1-Score", f"{f1_score*100:.2f}%")

    st.subheader("Confusion Matrix")

    # Replace with your actual confusion matrix values
    cm = [[450, 35],
          [28, 472]]

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Bleached", "Healthy"],
                yticklabels=["Bleached", "Healthy"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    st.pyplot(fig)
