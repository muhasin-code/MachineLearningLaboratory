# 🩺 Diabetes Prediction Using Random Forest

A complete machine learning project that predicts whether a patient is diabetic based on clinical measurements.  
This project includes full data preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, and evaluation.

---

## 📌 Project Overview

Diabetes is a chronic metabolic disease that often goes undetected in its early stages.  
This project builds an ML-based screening model using the **PIMA Indians Diabetes Dataset** and a **Random Forest Classifier**.

The objective is to detect high-risk individuals and support early diagnosis.

---

## 📂 Features of This Project

- Data cleaning & preprocessing  
- Exploratory Data Analysis (EDA)  
- Median imputation for impossible zero values  
- Train-test split with stratification  
- Baseline Random Forest model  
- Full hyperparameter tuning  
- GridSearchCV optimization  
- Final evaluation (Accuracy, Recall, F1 Score)  
- Feature importance analysis  
- Professional ML report  

---

## 📊 Dataset Information

The dataset contains **768 records** and **9 attributes**:

| Feature | Description |
|--------|-------------|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration |
| BloodPressure | Diastolic blood pressure |
| SkinThickness | Skin fold thickness |
| Insulin | 2-hour serum insulin |
| BMI | Body mass index |
| DiabetesPedigreeFunction | Family history score |
| Age | Age in years |
| Outcome | 1 = diabetic, 0 = non-diabetic |

---

## 🧹 Data Preprocessing

Certain medical values in the dataset appeared as **zeros**, which are medically invalid.  
The following features were fixed using **median imputation**:

- Glucose  
- BloodPressure  
- SkinThickness  
- Insulin  
- BMI  

Data was then split into:

- **80% training**
- **20% testing**
- Stratified sampling to maintain class balance

---

## 📈 Model Training & Tuning

We use **RandomForestClassifier** with iterative tuning:

### Hyperparameters tuned:
- n_estimators  
- max_depth  
- min_samples_split  
- min_samples_leaf  
- class_weight  

### Best parameters (via GridSearchCV):
```python
{
  'n_estimators': 200,
  'max_depth': 7,
  'min_samples_split': 10,
  'min_samples_leaf': 2,
  'class_weight': 'balanced'
}