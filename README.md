# ⚡ Energy Consumption Classification using SVM

## 📌 Project Overview

This project predicts whether household energy consumption is **Normal ⚡ or High 🔥** using a **Support Vector Machine (SVM)** model.
It also includes a **Streamlit-based web application** for real-time predictions.

---

## 🧠 Problem Statement

Energy consumption varies based on environmental conditions like temperature and humidity.
This project aims to classify consumption into:

* **Normal (< 300 Wh)**
* **High (≥ 300 Wh)**

---

## 📊 Dataset Description

| Feature    | Description                 |
| ---------- | --------------------------- |
| T1         | Living room temperature     |
| RH_1       | Living room humidity        |
| T2         | Kitchen temperature         |
| RH_2       | Kitchen humidity            |
| T_out      | Outdoor temperature         |
| RH_out     | Outdoor humidity            |
| Appliances | Energy consumption (Target) |

---

## ⚙️ Tech Stack

* Python
* Scikit-learn (SVM)
* Pandas, NumPy
* Streamlit (Web App)

---

## 🧠 Model Details

* Algorithm: Support Vector Machine (Linear SVM)
* Feature Scaling: StandardScaler
* Balanced Dataset using Downsampling
* Accuracy: 85.71% (update after running)
* Accuracy: XX%
* Precision: XX
* Recall: XX
* F1 Score: XX

---

## 🚀 Features

* Interactive web interface
* Real-time prediction
* Clean UI
* Binary classification (Normal / High)

---

## 📷 Application Preview

(Add screenshot here after running app)

---

## ▶️ How to Run Locally

### 1. Clone Repository

```bash
git clone https://github.com/your-username/energy-svm-app.git
cd energy-svm-app
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train Model

```bash
python train.py
```

### 5. Run Web App

```bash
streamlit run app.py
```

---

## ⚡ Energy Consumption Guide

* **Normal ⚡** → Less than 300 Wh
* **High 🔥** → Greater than or equal to 300 Wh

---

## 🎯 Future Improvements

* Add CSV upload for bulk prediction
* Deploy on cloud (Streamlit Cloud / Render)
* Add visualization dashboard
* Model explainability (SHAP)

---

## 👨‍💻 Author

**Himanshu Raj**

---

## ⭐ If you like this project

Give it a star ⭐ on GitHub!
