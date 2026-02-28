# PEGuard

PEGuard is a machine learning project focused on detecting **malicious Windows PE files**
using **static PE header features**.

The project is intentionally iterative: each stage builds on the previous one, with
clear analysis, validation, and justification for every modeling decision.

---

## Current Status

### Dataset
- Labeled Windows PE files (Benign / Malicious)
- Features extracted from PE headers

### Baseline Model
- **Model**: Logistic Regression
- **Pipeline**:
  - Median imputation for missing values
  - Feature standardization
  - End-to-end training using `sklearn.Pipeline`
- **Train/Test Split**:
  - Stratified split to preserve class balance

### Model Evaluation
- Accuracy
- Confusion matrix
- Classification report (precision, recall, F1-score)

---

## Validation & Diagnostics

### Cross-Validation
- Stratified 5-fold cross-validation
- Mean and standard deviation of accuracy reported
- Confirms model stability and low variance

### Bias–Variance Analysis
- Learning curves computed using increasing training set sizes
- Results indicate:
  - Low bias
  - Low variance
- Baseline model generalizes well

---

## Hyperparameter Tuning

- Regularization strength (**C**) tuned using `GridSearchCV`
- Best C selected based on cross-validation accuracy
- Tuned model evaluated again on held-out test set

---

## Cost-Sensitive Learning & Error Analysis

Because **false negatives (missed malware)** are more dangerous than false positives:

- Class weights were introduced for the *Malicious* class
- Multiple class-weight settings were evaluated
- Trade-off analyzed between:
  - False Negatives
  - False Positives
  - Malware recall

> Final models intentionally sacrifice some overall accuracy to significantly reduce
> false negatives — a correct and expected trade-off for security-critical systems.

---

## Key Takeaways So Far

- Accuracy alone is not a sufficient metric for malware detection
- Cost-sensitive learning materially improves safety by reducing missed malware
- Logistic Regression provides a strong, interpretable baseline

---

## Random Forest Robustness Analysis

Random Forest initially achieved:

- ROC-AUC ≈ 0.999  
- PR-AUC ≈ 0.998  
- Accuracy ≈ 99%

Feature importance analysis revealed heavy reliance on version/build metadata
(e.g., linker version, OS version, subsystem version).

To test robustness, controlled ablation experiments were performed:

- Removing linker features → slight performance drop (ROC-AUC ≈ 0.997)
- Removing all version metadata → significant drop:
  - ROC-AUC ≈ 0.982  
  - Accuracy ≈ 92%

### Insight

A substantial portion of performance is influenced by build-environment
metadata rather than purely structural malicious characteristics.

This suggests dataset-level bias and motivates further robustness testing
under metadata perturbation.

---

## Author
Sidh Virmani
