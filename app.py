import streamlit as st
import numpy as np
import cv2
import time
import os
import gdown
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# ----------------------------
# MODEL DOWNLOAD (NEW)
# ----------------------------

MODEL_PATH = "coral_model_mobilenet.h5"
FILE_ID = "1mkQEpRIBSBF-H10mKj3_nbF7TbjjHSxg"

if not os.path.exists(MODEL_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)

# ----------------------------
# LOAD MODEL (FIXED)
# ----------------------------

@st.cache_resource
def load_model_fixed():

    IMG_SIZE = 224

    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights=None
    )

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=base_model.input, outputs=output)

    model.load_weights(MODEL_PATH)

    return model


model = load_model_fixed()

# ----------------------------
# TITLE
# ----------------------------

st.title("🌊 Coral Reef Health Monitoring System")
st.write("Upload a coral image to detect whether it is healthy or bleached.")

# ----------------------------
# TABS
# ----------------------------

tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Model Performance", "ℹ️ About"])

# ----------------------------
# TAB 1 — PREDICTION
# ----------------------------

with tab1:

    uploaded_file = st.file_uploader("Upload Coral Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        start = time.time()
        prediction = model.predict(img, verbose=0)[0][0]
        end = time.time()

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if prediction > 0.5:
            confidence = prediction * 100
            st.error(f"Bleached Coral ({confidence:.2f}%)")
        else:
            confidence = (1 - prediction) * 100
            st.success(f"Healthy Coral ({confidence:.2f}%)")

        st.info(f"Inference Time: {round(end - start, 4)} sec")
        st.warning("This is an AI-based prediction and may not be 100% accurate.")

# ----------------------------
# TAB 2 — PERFORMANCE
# ----------------------------

with tab2:

    st.subheader("Model Performance")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "85%")
    col2.metric("Precision", "85%")
    col3.metric("Recall", "85%")
    col4.metric("F1-Score", "85%")

    st.subheader("Confusion Matrix")

    cm = [[426, 59],
          [86, 414]]

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Bleached", "Healthy"],
                yticklabels=["Bleached", "Healthy"])

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    st.pyplot(fig)

# ----------------------------
# TAB 3 — ABOUT
# ----------------------------

with tab3:

    st.subheader("About Project")

    st.write("""
    This project detects coral bleaching using deep learning.

    A MobileNetV2 transfer learning model is used to classify coral images into:
    - Healthy Coral
    - Bleached Coral

    Achieved accuracy: ~85%
    """)

    st.subheader("Technologies")
    st.markdown("""
    - Python  
    - TensorFlow / Keras  
    - OpenCV  
    - Streamlit  
    """)

    st.subheader("Future Scope")
    st.markdown("""
    - IoT sensor integration  
    - Satellite monitoring  
    - Mobile app development  
    """)

    st.success("Academic Project - Coral Reef Monitoring")