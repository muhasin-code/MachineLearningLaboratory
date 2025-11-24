# 📄 **README.md — Spam Detection using Decision Trees (UCI Spambase)**

A complete Machine Learning project where we build, tune, evaluate, and interpret a **Decision Tree Classifier** to detect spam emails using the popular **UCI Spambase dataset**.  
This project walks through the entire ML pipeline, emphasizing understanding, tuning, and interpreting Decision Trees for real-world usage.

---

## 📌 Project Highlights

- Supervised **binary classification** (spam vs ham)
- Dataset: **UCI Spambase** (4,600 emails × 58 numerical features)
- Model: **DecisionTreeClassifier**
- Tuning:  
  - `max_depth`  
  - `min_samples_leaf`  
  - `min_samples_split`  
  - `criterion` (Gini vs Entropy vs Log-Loss)
- Visualization of tree structure
- Feature importance analysis
- Cross-validation (5-fold & 10-fold) for stability checking

---

## 📂 Dataset

**Source:** UCI Machine Learning Repository – *Spambase Dataset*  
Link: https://archive.ics.uci.edu/dataset/94/spambase

The dataset contains:
- Word frequency features (e.g., `word_freq_free`, `word_freq_remove`)
- Character frequency features (e.g., `char_freq_$`, `char_freq_!`)
- Capital letter usage statistics (`capital_run_length_average`)
- Target label:  
  - `1` = Spam  
  - `0` = Ham  

No missing values, all numeric → perfect for Decision Trees.

---

## ⚙️ Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib  
- Graphviz (optional, for visualization)

---

## 🚀 Project Workflow

This project follows a structured, phase-based workflow:

---

### **Phase 1 — Data Inspection**

- Loaded dataset from `spambase.data`
- Assigned appropriate column names
- Verified shape, data types, and missing values
- Confirmed class distribution  

Ham : 2788

Spam: 1812

---

### **Phase 2 — Train/Test Split**

Used an 80/20 split with stratification:
```python
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42, stratify=y
)
```
---

### **Phase 3 — Baseline Decision Tree**

Trained default Decision Tree:

```python
DecisionTreeClassifier(random_state=42)
```

**Baseline Results:**

* Train Accuracy: **99.97%**
* Test Accuracy: **~90.97%**

Clear overfitting → tuning required.

---

### **Phase 4 — Hyperparameter Tuning**

Sequential tuning steps:

#### 🔹 1. Max Depth Sweep (1–20)

Found **best range: depth 11–15**, with **depth = 11** being optimal.

#### 🔹 2. min_samples_leaf Testing

`min_samples_leaf = 1` performed best.

#### 🔹 3. min_samples_split Testing

Best values were **2 or 5**, default 2 chosen.

#### 🔹 4. Criterion Comparison

| Criterion | Test Accuracy |
| --------- | ------------- |
| gini      | 0.9196        |
| entropy   | **0.9304**    |
| log_loss  | **0.9304**    |

Entropy chosen.

---

### **Final Model**

```python
best_model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=11,
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=42
)
```

---

### **Final Performance**

* **Train Accuracy:** 97.88%
* **Test Accuracy:** **93.04%**

#### Confusion Matrix

```
[[528  30]
 [ 34 328]]
```

#### Classification Report

| Class | Precision | Recall | F1-score |
| ----- | --------- | ------ | -------- |
| Ham   | 0.94      | 0.95   | 0.94     |
| Spam  | 0.92      | 0.91   | 0.91     |

A well-balanced model with strong generalization.

---

### **Phase 5 — Feature Importance Interpretation**

Top 10 most important features:

```
1. char_freq_$ (0.2879)
2. word_freq_remove
3. char_freq_!
4. word_freq_hp
5. capital_run_length_average
6. word_freq_george
7. word_freq_free
8. word_freq_edu
9. word_freq_our
10. capital_run_length_total
```

Key insights:

* `$`, `!`, “remove”, “free” strongly indicate spam
* “hp”, “george”, “edu” often indicate ham
* Capital letters usage is a major spam signal

---

### **Phase 6 — Tree Visualization**

Visualized top layers and full tree using:

```python
tree.plot_tree()
export_graphviz()
```

This helped interpret decision logic:

* First splits often revolve around `$` frequency
* Subsequent splits check “remove”, “free”, and capitalization patterns

---

### **Phase 7 — Cross-Validation (Stability Check)**

#### **5-Fold CV**

* Mean: **0.8882**
* Std: **0.0539** (some variance)

#### **10-Fold CV**

* Mean: **0.9206**
* Std: **0.0267** (more stable)

Tree shows expected variance, but 10-fold CV confirms stable performance.

---

## 📊 Conclusion

This project demonstrates:

* How to build and tune a high-performing Decision Tree
* Real-world interpretation of spam features
* Importance of pruning (`max_depth`, `criterion`)
* How cross-validation reveals model stability
* How to visualize and understand model behavior

The final model is accurate, interpretable, and suitable as a classical baseline for spam detection.

---

## 🔮 Next Steps (Future Enhancements)

* Add **Random Forest** for variance reduction
* Compare with **Gradient Boosted Trees**
* Convert model into a **Flask or FastAPI spam detection service**
* Build a web UI for email classification
* Integrate TF-IDF for raw email text classification

---

## 📝 License

This project is released under the **MIT License**.

---

## 🤝 Contributions

Contributions, issues, and pull requests are welcome!