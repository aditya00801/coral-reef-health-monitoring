# 🌊 Coral Reef Health Monitoring System

## 📌 Overview
The **Coral Reef Health Monitoring System** is a Deep Learning-based web application designed to detect coral bleaching using a Convolutional Neural Network (CNN).

Coral bleaching is a serious environmental issue caused by rising sea temperatures and climate change. This system aims to assist in early detection and monitoring of reef health through AI-driven image classification.

---

## 🚀 Live Demo
🔗 Streamlit Web App:  
[Add Your Streamlit Link Here]

---

## 🧠 Model Architecture

- Model Type: Convolutional Neural Network (CNN)
- Input Shape: 224 × 224 × 3 (RGB Image)
- Output Classes:
  - Bleached Coral
  - Healthy Coral
- Activation: Softmax (Final Layer)
- Loss Function: Categorical Crossentropy
- Optimizer: Adam

---

## 📊 Model Performance

| Metric      | Score |
|-------------|--------|
| Accuracy    | 92%   |
| Precision   | 91%   |
| Recall      | 93%   |
| F1-Score    | 92%   |

*Note: Model is currently under continuous optimization.*

---

## 📁 Dataset Information

- Training Samples: 7,384 images
- Validation Samples: 985 images
- Classes:
  - Bleached Corals
  - Healthy Corals

Images were resized to 224×224 and normalized before training.

---

## 🛠 Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Streamlit
- Google Drive (Model Hosting)
- GitHub (Version Control)

## 📦 Project Structure

```
coral-reef-health-monitoring/
│
├── app.py                # Streamlit Web Application
├── train_model.py        # CNN Training Script
├── test_model.py         # Model Testing Script
├── requirements.txt      # Project Dependencies
├── README.md             # Project Documentation
└── .gitignore
```

---

## 🎯 Objectives

- Detect coral bleaching using AI.
- Build an accurate CNN classification model.
- Deploy the model as a 24×7 web application.
- Provide a user-friendly dashboard for prediction and performance visualization.

---

## 🚀 Future Enhancements

- Integration with real-time satellite data
- IoT-based environmental parameter monitoring
- Multi-class coral disease detection
- Mobile application version
- Live marine ecosystem analytics dashboard

---

## 👥 Team Members

- ADITYA KUSHWAHA
- ADITYA KUMAR
- ANKIT KUMAR
- AMAN KUMAR SINGH

---

## 📌 Project Type

Academic Minor Project  
Domain: Artificial Intelligence & Deep Learning  
Deployment: Streamlit Cloud (24×7)

---

⚠ Disclaimer: This model is under continuous optimization. Performance may improve in future updates.
