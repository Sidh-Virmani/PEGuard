# PEGuard – Malware Detection using Machine Learning

## 📌 Overview
This project focuses on detecting whether a Windows executable (PE file) is **malicious or benign** using machine learning models trained on static binary features.

The goal is to:
- Compare multiple ML models under **fair (balanced)** conditions
- Evaluate performance under **real-world (imbalanced)** conditions
- Identify which models are most reliable for **security-critical applications**

---

## 📊 Datasets Used

### 1. Balanced Dataset
- Source: https://www.kaggle.com/datasets/adilalzada/dataset-for-my-honours
- Distribution:
  - Benign: ~54.7%
  - Malicious: ~45.3%

### 2. Real-World Dataset (Imbalanced)
- Source: https://www.kaggle.com/datasets/dasarijayanth/pe-header-data
- Distribution:
  - Benign: ~70%
  - Malicious: ~30%

---

## ⚙️ Models Implemented

All models are kept **vanilla (default settings)** for fair comparison:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Support Vector Machine (SVM)
- Gradient Boosting
- AdaBoost
- Multi-Layer Perceptron (MLP)
- XGBoost

---

## 📈 Evaluation Metrics

Models are evaluated using:

- Accuracy
- Precision
- Recall (**critical for malware detection**)
- F1 Score
- ROC-AUC
- False Positive Rate (FPR)
- False Negative Rate (FNR)
- Specificity
- Balanced Accuracy

> Note: In malware detection, **False Negatives are the most critical**, as they represent malicious files classified as safe.

---

## Author
Sidh Virmani


