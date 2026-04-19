# 🌊 Coral Reef Health Monitoring System

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge\&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange?style=for-the-badge\&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red?style=for-the-badge\&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

---

## 📌 Overview

The **Coral Reef Health Monitoring System** is an AI-powered web application designed to detect coral bleaching using a **Convolutional Neural Network (CNN)**.

Coral reefs are vital to marine ecosystems but are increasingly threatened by **climate change and rising ocean temperatures**. This project leverages **Deep Learning** to provide automated, accurate, and scalable coral health analysis.

---

## 🚀 Live Demo

🔗 **Try the App:**
https://coral-reef-health-monitoring-a4dcmczx2uqg6sen2huygv.streamlit.app/

---

## 🖼️ Application Preview

*(Add screenshots here for better impression)*

![App Screenshot](screenshots/app1.png)
![Prediction Result](screenshots/app2.png)

---

## 🧠 Model Architecture

* **Model Type:** CNN (Transfer Learning)
* **Input Size:** 224 × 224 × 3
* **Classes:**

  * 🔴 Bleached Coral
  * 🟢 Healthy Coral
* **Activation:** Softmax
* **Loss Function:** Categorical Crossentropy
* **Optimizer:** Adam

---

## 📊 Model Performance

| Metric    | Score |
| --------- | ----- |
| Accuracy  | 92%   |
| Precision | 91%   |
| Recall    | 93%   |
| F1-Score  | 92%   |

> ⚠ Model is under continuous optimization and improvements are ongoing.

---

## 📁 Dataset Details

* **Total Images:** ~8,300+
* **Training Samples:** 7,384
* **Validation Samples:** 985
* **Classes:** Healthy vs Bleached Coral
* **Preprocessing:**

  * Resizing → 224×224
  * Normalization
  * Data splitting

---

## 🛠 Tech Stack

| Category         | Tools Used         |
| ---------------- | ------------------ |
| Language         | Python             |
| Deep Learning    | TensorFlow / Keras |
| Image Processing | OpenCV             |
| Web Framework    | Streamlit          |
| Deployment       | Streamlit Cloud    |
| Storage          | Google Drive       |
| Version Control  | GitHub             |

---

## 📦 Project Structure

```
coral-reef-health-monitoring/
│
├── app.py                # Streamlit application
├── train_model.py        # Model training script
├── test_model.py         # Model evaluation
├── requirements.txt      # Dependencies
├── README.md             # Documentation
└── .gitignore
```

---

## ⚙️ Installation & Setup

### 🔹 Clone Repository

```bash
git clone https://github.com/your-username/coral-reef-health-monitoring.git
cd coral-reef-health-monitoring
```

### 🔹 Install Dependencies

```bash
pip install -r requirements.txt
```

### 🔹 Run Application

```bash
streamlit run app.py
```

---

## 🎯 Objectives

* Automate coral bleaching detection using AI
* Build a high-accuracy CNN model
* Provide real-time predictions via web app
* Support marine ecosystem conservation

---

## 🚀 Future Enhancements

* 🌐 Real-time satellite data integration
* 🤖 IoT-based environmental monitoring
* 📱 Mobile app deployment
* 🧠 Multi-class coral disease detection
* 📊 Advanced analytics dashboard

---

## 👥 Team Members

* **Aditya Kushwaha**
* **Aditya Kumar**
* **Ankit Kumar**
* **Aman Kumar Singh**

---

## 📌 Project Details

* **Type:** Academic Minor Project
* **Domain:** Artificial Intelligence & Deep Learning
* **Deployment:** Streamlit Cloud (24×7)

---

## ⚠ Disclaimer

This project is developed for academic and research purposes.
Performance may vary depending on dataset quality and environmental conditions.

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!
