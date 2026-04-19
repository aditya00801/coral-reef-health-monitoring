import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import os
import gdown
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime

# ===============================
# CONFIG
# ===============================
FILE_ID = "1AMHjMG81IsNlcsFB_TaV4DwA0u8BpCsq"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"
MODEL_PATH = "model.keras"

IMG_SIZE = 224
CLASS_NAMES = ["Bleached Coral", "Healthy Coral"]

st.set_page_config(page_title="Coral Health Monitor", layout="wide")

# ===============================
# MODEL LOAD
# ===============================
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

@st.cache_resource
def load_cnn_model():
    download_model()
    return load_model(MODEL_PATH)

model = load_cnn_model()

# ===============================
# PREPROCESS
# ===============================
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)
    image = preprocess_input(image)
    return np.expand_dims(image, axis=0)

# ===============================
# PDF REPORT
# ===============================
def create_pdf(label, confidence, health, risk):
    file_path = "coral_report.pdf"
    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()

    content = []
    content.append(Paragraph("Coral Reef Health Report", styles["Title"]))
    content.append(Spacer(1, 20))
    content.append(Paragraph(f"Generated: {datetime.now()}", styles["Normal"]))
    content.append(Spacer(1, 10))
    content.append(Paragraph(f"Prediction: {label}", styles["Normal"]))
    content.append(Paragraph(f"Confidence: {confidence:.2f}%", styles["Normal"]))
    content.append(Paragraph(f"Health Score: {health:.2f}%", styles["Normal"]))
    content.append(Paragraph(f"Risk Level: {risk}", styles["Normal"]))

    doc.build(content)
    return file_path

# ===============================
# TABS
# ===============================
tab1, tab2 = st.tabs(["🔍 Prediction", "ℹ About"])

# ===============================
# PREDICTION TAB
# ===============================
with tab1:
    st.title("🌊 Coral Reef Health Monitoring System")
    st.write("Upload a coral image to analyze reef health.")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing..."):
            processed = preprocess_image(image)
            prediction = model.predict(processed)[0]

        confidence = float(np.max(prediction)) * 100
        class_index = np.argmax(prediction)
        label = CLASS_NAMES[class_index]

        # Health logic
        health = confidence if label == "Healthy Coral" else 100 - confidence

        # Risk level
        if health > 80:
            risk = "Low Risk"
            st.success("🟢 Low Risk")
        elif health > 50:
            risk = "Moderate Risk"
            st.warning("🟡 Moderate Risk")
        else:
            risk = "High Risk"
            st.error("🔴 High Risk")

        with col2:
            st.metric("Prediction", label)
            st.metric("Confidence", f"{confidence:.2f}%")
            st.metric("Health Score", f"{health:.2f}%")
            st.progress(int(health))

        # Graph
        st.subheader("Prediction Confidence")
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, prediction * 100)
        ax.set_ylabel("Confidence (%)")
        st.pyplot(fig)

        # PDF
        pdf_path = create_pdf(label, confidence, health, risk)
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download Report", f, file_name="coral_report.pdf")

# ===============================
# ABOUT TAB
# ===============================
with tab2:
    st.title("ℹ About Project")

    st.markdown("""
    ### 🌊 Coral Reef Health Monitoring System

    This system uses **EfficientNetB0 (Deep Learning)** to detect coral health.

    ### 🎯 Objective
    - Detect coral bleaching
    - Assist marine conservation
    - Provide automated analysis

    ### 🧠 Model Info
    - Architecture: EfficientNetB0
    - Method: Transfer Learning
    - Accuracy: ~98%
    - Input Size: 224x224

    ### 📊 Classes
    - 🟢 Healthy Coral
    - 🔴 Bleached Coral

    ### ⚙️ Features
    - Image classification
    - Risk level detection
    - Confidence graph
    - PDF report generation

    ### 🛠 Tech Stack
    - Python
    - TensorFlow / Keras
    - Streamlit

    ---
    Developed as a B.Tech Machine Learning Project
    """)
