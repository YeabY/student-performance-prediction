# 🎓 Student Performance Prediction

Predict student final grades using Machine Learning models based on lifestyle, study habits, and demographic data.

This project aims to help educators identify students at risk and intervene early by analyzing a variety of academic and social factors.

---

## 🚀 Features

- Predict final grades (G3) using supervised ML models  
- Interactive GUI using Streamlit for real-time prediction  
- Exploratory Data Analysis (EDA) and feature importance visualization  
- Supports early identification of underperforming students  
- Models implemented: Linear Regression, Random Forest, Gradient Boosting  

---

## 📊 Dataset

- **Source**: UCI Machine Learning Repository  
- **Name**: Student Performance Dataset  
- **Link**: [UCI Student Performance Dataset](https://archive.ics.uci.edu/ml/datasets/student+performance)  
- **Instances**: 395 students  
- **Features**: 33 attributes including:  
  - Demographics: Age, gender, address, family size  
  - Lifestyle: Free time, internet access, health, absences  
  - Academics: Study time, failures, G1 & G2 scores  
- **Target**: G3 (final grade, range: 0–20)  

---

## 🧠 Models & Performance

| Model              | MAE  | RMSE | R²   |
|-------------------|------|------|------|
| Linear Regression | 2.5  | 3.4  | 0.31 |
| Random Forest     | 1.8  | 2.7  | 0.64 |
| Gradient Boosting | 1.6  | 2.4  | 0.67 ✅ |

Gradient Boosting performed best and is used in the final application.

---

## 💻 Streamlit GUI

A simple Streamlit app is provided to interactively predict student grades.

### To Run:

```bash
pip install -r requirements.txt
streamlit run app.py
