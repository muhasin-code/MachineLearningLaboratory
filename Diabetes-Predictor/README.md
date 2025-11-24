# **Diabetes Prediction Using Random Forest**


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
```

---

## 🧪 Final Model Performance

| Metric        | Score |
| ------------- | ----- |
| **Accuracy**  | 0.76  |
| **Precision** | 0.63  |
| **Recall**    | 0.74  |
| **F1 Score**  | 0.68  |

### Confusion Matrix

```
[[77 23]
 [14 40]]
```

The model focuses on **recall**, as missing a diabetic patient is more dangerous than a false alarm.

---

## 🔍 Feature Importance

| Feature                  | Importance |
| ------------------------ | ---------- |
| Glucose                  | 0.327      |
| BMI                      | 0.179      |
| Age                      | 0.137      |
| DiabetesPedigreeFunction | 0.102      |
| Insulin                  | 0.064      |
| Pregnancies              | 0.064      |
| SkinThickness            | 0.064      |
| BloodPressure            | 0.062      |

**Glucose** is the most influential feature, which aligns with clinical expectations.

---

## 🚀 How to Run the Project

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Notebook

Open `diabetes_prediction.ipynb` in Jupyter or Google Colab.

### 3. (Optional) Run the Model as a Web App

A Streamlit or Flask deployment can be added on request.

---

## 📌 Future Enhancements

* Add SHAP/LIME explainability
* Deploy using Streamlit
* Build a REST API using Flask
* Add external validation datasets
* Use advanced models (XGBoost, LightGBM)

---

## 👨‍💻 Author

**Muhammed Muhasin K**

---

## 📜 License

This project is open-source under the MIT License.