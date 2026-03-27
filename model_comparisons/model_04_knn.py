import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import *

from common_preprocessing import get_train_test_data


DATASET_NAME = "real_world"

X_train, X_test, y_train, y_test = get_train_test_data(DATASET_NAME)

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", KNeighborsClassifier())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

results = {
    "Model": "KNN",
    "Dataset": DATASET_NAME,
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, zero_division=0),
    "Recall": recall_score(y_test, y_pred, zero_division=0),
    "F1 Score": f1_score(y_test, y_pred, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, y_prob),
    "False Positive Rate": fp/(fp+tn),
    "False Negative Rate": fn/(fn+tp),
    "TP": tp, "TN": tn, "FP": fp, "FN": fn
}

pd.DataFrame([results]).to_csv(f"../results/knn_{DATASET_NAME}.csv", index=False)