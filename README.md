# 🫀 Cardiovascular Disease Prediction App

This project is a machine learning-powered web application that predicts the likelihood of cardiovascular disease based on patient input data. It uses a Decision Tree Classifier trained on a medical dataset and deployed with **Streamlit** for easy interaction.

---

## 📁 Dataset Overview

The dataset used is a commonly referenced **heart disease dataset**, typically derived from the Cleveland Clinic Heart Disease dataset.

### 🔍 Attribute Explanations:

| Feature     | Description |
|-------------|-------------|
| **age**     | Age of the patient (years) |
| **sex**     | Gender (0 = Female, 1 = Male) |
| **cp**      | Chest pain type:<br>0 = Typical angina<br>1 = Atypical angina<br>2 = Non-anginal pain<br>3 = Asymptomatic |
| **trestbps**| Resting blood pressure (in mm Hg) |
| **chol**    | Serum cholesterol (in mg/dl) |
| **fbs**     | Fasting blood sugar > 120 mg/dl (1 = True; 0 = False) |
| **restecg** | Resting electrocardiographic results:<br>0 = Normal<br>1 = ST-T wave abnormality<br>2 = Left ventricular hypertrophy |
| **thalach** | Maximum heart rate achieved |
| **exang**   | Exercise induced angina (1 = Yes; 0 = No) |
| **oldpeak** | ST depression induced by exercise relative to rest |
| **slope**   | Slope of the peak exercise ST segment:<br>0 = Upsloping<br>1 = Flat<br>2 = Downsloping |
| **ca**      | Number of major vessels colored by fluoroscopy (0–3) |
| **thal**    | Thalassemia:<br>1 = Normal<br>2 = Fixed defect<br>3 = Reversible defect |
| **target**  | Diagnosis of heart disease (1 = disease, 0 = no disease) |

---

## 🧠 Model Training

### ⚙️ Algorithm:
A **Decision Tree Classifier** was trained on the dataset.

### 🔬 Preprocessing:
- Split: 80% training / 20% testing
- Scaled features using `StandardScaler` from `sklearn`

### 📈 Performance:

| Metric      | Value |
|-------------|-------|
| **Accuracy**| *~82%* (varies slightly due to random split) |
| **Model**   | `DecisionTreeClassifier()` from scikit-learn |

---

## 🖥️ Streamlit Web App

Deployed link: https://cardiovascular-disease-prediction-system.streamlit.app/

### 🚀 How to Run in local machine

```bash
pip install -r requirements.txt
streamlit run app.py
