# cnn_realworld_excel.py

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import *

from common_preprocessing import get_train_test_data

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam


def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "Model": "CNN_1D",
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "False Positive Rate": fp / (fp + tn),
        "False Negative Rate": fn / (fn + tp),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Specificity": tn / (tn + fp),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred)
    }


def build_cnn(input_shape):
    model = Sequential([
        Conv1D(64, 3, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv1D(32, 3, activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def main():
    X_train, X_test, y_train, y_test = get_train_test_data("real_world")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # reshape for CNN
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = build_cnn((X_train.shape[1], 1))

    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=256,
        verbose=1
    )

    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob > 0.5).astype(int)

    results = compute_metrics(y_test, y_pred, y_prob)

    df = pd.DataFrame([results])

    os.makedirs("results", exist_ok=True)
    df.to_excel("results/CNN_realworld_results.xlsx", index=False)

    print("✅ Saved: results/CNN_realworld_results.xlsx")


if __name__ == "__main__":
    main()