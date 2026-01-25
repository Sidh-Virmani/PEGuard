# PEGuard

PEGuard is a work-in-progress machine learning project for detecting **malicious Windows PE files** using **static PE header features**.

The goal is to iteratively build, analyze, and improve malware detection models while revising core ML concepts (preprocessing, evaluation, debugging, model selection).

---

## Current Status (Baseline)

- Dataset: Labeled Windows PE files (Benign / Malicious)
- Features: Static PE header attributes
- Model: Logistic Regression (baseline)
- Preprocessing:
  - Removed non-informative identifiers
  - Handled missing values
  - Feature scaling
- Evaluation:
  - Train/Test split
  - Accuracy, confusion matrix, classification report

This baseline serves as a reference point for further improvements.

---

## Planned Improvements

- Better feature handling (hex → numeric, richer PE features)
- Cross-validation
- Alternative models (tree-based, ensembles, neural networks)
- Error analysis and model debugging
- Feature importance and interpretability

---

---

## Author
Sidh Virmani


