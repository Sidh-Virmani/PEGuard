import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten, Dropout, BatchNormalization, Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from common_preprocessing import get_train_test_data


def compute_metrics(model_name, y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) != 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "False Positive Rate": fpr,
        "False Negative Rate": fnr,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Specificity": specificity,
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred)
    }


def build_cnn(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(64, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        Conv1D(32, kernel_size=3, activation="relu", padding="same"),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    X_train_raw, X_test_raw, y_train, y_test = get_train_test_data("real_world")

    # scaled version for models that need scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    results = []

    # 1. Logistic Regression
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    results.append(compute_metrics("Logistic Regression", y_test, y_pred, y_prob))

    # 2. Decision Tree
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_raw, y_train)
    y_pred = model.predict(X_test_raw)
    y_prob = model.predict_proba(X_test_raw)[:, 1]
    results.append(compute_metrics("Decision Tree", y_test, y_pred, y_prob))

    # 3. Random Forest
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train_raw, y_train)
    y_pred = model.predict(X_test_raw)
    y_prob = model.predict_proba(X_test_raw)[:, 1]
    results.append(compute_metrics("Random Forest", y_test, y_pred, y_prob))

    # 4. KNN
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    results.append(compute_metrics("KNN", y_test, y_pred, y_prob))

    # 5. Naive Bayes
    model = GaussianNB()
    model.fit(X_train_raw, y_train)
    y_pred = model.predict(X_test_raw)
    y_prob = model.predict_proba(X_test_raw)[:, 1]
    results.append(compute_metrics("Naive Bayes", y_test, y_pred, y_prob))

    # 6. SVM
    model = SVC(probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    results.append(compute_metrics("SVM", y_test, y_pred, y_prob))

    # 7. Gradient Boosting
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train_raw, y_train)
    y_pred = model.predict(X_test_raw)
    y_prob = model.predict_proba(X_test_raw)[:, 1]
    results.append(compute_metrics("Gradient Boosting", y_test, y_pred, y_prob))

    # 8. AdaBoost
    model = AdaBoostClassifier(random_state=42, n_estimators=100)
    model.fit(X_train_raw, y_train)
    y_pred = model.predict(X_test_raw)
    y_prob = model.predict_proba(X_test_raw)[:, 1]
    results.append(compute_metrics("AdaBoost", y_test, y_pred, y_prob))

    # 9. MLP
    model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    results.append(compute_metrics("MLP", y_test, y_pred, y_prob))

    # 10. XGBoost
    model = XGBClassifier(
        random_state=42,
        n_estimators=100,
        eval_metric="logloss"
    )
    model.fit(X_train_raw, y_train)
    y_pred = model.predict(X_test_raw)
    y_prob = model.predict_proba(X_test_raw)[:, 1]
    results.append(compute_metrics("XGBoost", y_test, y_pred, y_prob))

    # 11. CNN_1D
    X_train_cnn = np.array(X_train_scaled).reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
    X_test_cnn = np.array(X_test_scaled).reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

    cnn = build_cnn((X_train_cnn.shape[1], 1))

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    cnn.fit(
        X_train_cnn,
        y_train,
        validation_split=0.1,
        epochs=10,
        batch_size=256,
        verbose=1,
        callbacks=[early_stop]
    )

    y_prob = cnn.predict(X_test_cnn, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    results.append(compute_metrics("CNN_1D", y_test, y_pred, y_prob))

    df = pd.DataFrame(results)

    df = df[[
        "Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC",
        "False Positive Rate", "False Negative Rate",
        "TP", "TN", "FP", "FN",
        "Specificity", "Balanced Accuracy"
    ]]

    os.makedirs("../results", exist_ok=True)
    output_path = "../results/final_real_world_all_models.xlsx"
    df.to_excel(output_path, index=False)

    print(f"\nSaved successfully: {output_path}")
    print(df)


if __name__ == "__main__":
    main()