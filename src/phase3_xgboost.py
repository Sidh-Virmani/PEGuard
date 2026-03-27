"""
XGBoost (Boosted Trees)

Why XGBoost after Random Forest?
- Random Forest builds many independent trees and averages their predictions.
- XGBoost builds trees sequentially such that each new tree focuses on correcting mistakes made so far.
- This "error-correction" gives better performance

What is XGBoost?
- Start with a simple model.
- Train a small tree to correct current errors.
- Repeat: each new tree adds targeted improvements.
- Final prediction is the sum of all trees outputs.

Goal of this phase:
- Determine whether boosted trees can improve deployability over both Logistic Regression and Random Forest.
"""