import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_balanced_dataset():
    csv_path = "../dataset/PE_Dataset_Labeled.csv"
    df = pd.read_csv(csv_path)

    # Shuffle because dataset is ordered by class
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Drop leakage / useless columns
    drop_cols = [col for col in ["Unnamed: 0", "File_Name"] if col in df.columns]

    # Target
    y = df["Label"].map({"Benign": 0, "Malicious": 1})

    # Features
    X = df.drop(columns=drop_cols + ["Label"], errors="ignore")

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    return X, y


def load_real_world_dataset():
    csv_path = "../dataset/PE_Header_Data.csv"
    df = pd.read_csv(csv_path, sep="|")

    # Shuffle because dataset is ordered by class
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Target: legitimate=1 means benign, so convert to malware target
    y = df["legitimate"].map({1: 0, 0: 1})

    # Features
    X = df.drop(columns=["legitimate", "Name", "md5"], errors="ignore")

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    return X, y


def get_dataset(dataset_name="balanced"):
    if dataset_name == "balanced":
        return load_balanced_dataset()
    elif dataset_name == "real_world":
        return load_real_world_dataset()
    else:
        raise ValueError("dataset_name must be 'balanced' or 'real_world'")


def get_train_test_data(dataset_name="balanced"):
    X, y = get_dataset(dataset_name)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test