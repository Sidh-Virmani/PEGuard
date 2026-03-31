# PEGuard – Malware Detection using Machine Learning & Deep Learning

## Overview

PEGuard is a machine learning-based system for detecting whether a Windows Portable Executable (PE) file is **malicious or benign** using static binary features.

The project evaluates multiple classical ML models and deep learning models on two datasets to understand performance, robustness, and real-world applicability.

---

## Objectives

* Classify PE files as **Malicious (1)** or **Benign (0)**
* Compare performance across multiple models
* Evaluate impact of dataset imbalance
* Analyze reliability using multiple evaluation metrics

---

## Datasets

Two datasets are used:

1. **Balanced Dataset**

   * ~54% Benign, ~46% Malicious
   * Used for fair comparison

2. **Real-World Dataset**

   * ~70% Benign, ~30% Malicious
   * Simulates real-world class imbalance

---

## Features

* Static PE file features (header metadata, structure-related attributes)
* Only **numeric features** used
* No raw binary or dynamic analysis

---

## Models Used

### Machine Learning Models

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors (KNN)
* Naive Bayes
* Support Vector Machine (SVM)
* Gradient Boosting
* AdaBoost
* XGBoost

### Deep Learning Models

* Multi-Layer Perceptron (MLP)
* 1D Convolutional Neural Network (CNN)

---

## Evaluation Metrics

Each model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* False Positive Rate (FPR)
* False Negative Rate (FNR)
* Confusion Matrix (TP, TN, FP, FN)
* Specificity
* Balanced Accuracy

---

## Project Structure

```
PEGUARD/
│
├── dataset/
│   ├── PE_Dataset_Labeled.csv
│   └── PE_Header_Data.csv
│
├── model_comparisons/
│   ├── common_preprocessing.py
│   ├── run_all_models_balanced.py
│   └── run_all_models_real_world.py
│
├── results/
│   ├── final_balanced_all_models.xlsx
│   └── final_real_world_all_models.xlsx
│
├── requirements.txt
└── README.md
```

---

## How to Run

### Step 1: Activate Virtual Environment

```bash
.\.venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run Models

For balanced dataset:

```bash
python model_comparisons/run_all_models_balanced.py
```

For real-world dataset:

```bash
python model_comparisons/run_all_models_real_world.py
```

---

## Output

The scripts generate:

* `results/final_balanced_all_models.xlsx`
* `results/final_real_world_all_models.xlsx`

Each file contains performance comparison of all models.

---

## Author

Sidh Virmani

