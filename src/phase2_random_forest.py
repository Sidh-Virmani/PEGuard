"""
Random Forest (Model Comparison)

Why Random Forest
    1) Logistic Regression learns a linear decision boundary.
    2) Malware detection may have non linear patterns and feature interactions
    (e.g., "this value is suspicious only when combined with another field's particular value").
    
What is a decision tree?
Its like follows:
    If Feature_A > 10:
        If Feature_B < 5:
            → Malware
        Else:
            → Benign
    Else:
        → Benign

What is Random Forest?
    1)A single decision tree learns a set of if else rules on features. It forms a non-linear decision boundary. 
    It is similar to a flowchart that splits data based on feature values to reach a decision by the end
    2) Random Forest trains many trees on different random subsets of the data and features.
    3) Final predictions come voting of the trees, which improves precision and reduces overfitting compared to a single tree.

Goal of this phase:
    - Evaluate whether Random Forest can improve the security operating point achieved by the baseline model.
    - We will compare models using ROC/PR behavior and threshold decision metrics.
    - Our baseline reference operating point is threshold = 0.6
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
#This time we don't need to scale features for tree based models, which is because trees are not sensitive to feature scales. 
# They make splits based on feature values whose relative ordering does not change with scaling. Hence dont need StandardScaler

CSV_PATH = "../dataset/PE_Dataset_Labeled.csv"

df = pd.read_csv(CSV_PATH)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df = df.drop(columns=["Unnamed: 0", "File_Name"])

y = df["Label"]
X = df.drop(columns=["Label"])

X = X.select_dtypes(include=[np.number]) #Keeping just the numeric features for this phase, as tree based models can handle them well and we want to focus on model comparison. We can explore adding categorical features in future phases if needed.

# Identical test train split as before to ensure fair comparison of models
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7, stratify=y
)

# Build random forest pipeline
rf_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=200,       #n_estimators is the number of trees in the forest = 200 for now, we can tune this later
        random_state=42,        #obv for reproducibility
        n_jobs=-1               #use all CPU cores for faster training (idk about this but library has this parameter)
    ))
])
    
    
#Train the model
rf_pipeline.fit(X_train, y_train)
    
    
rf_pred = rf_pipeline.predict(X_test)

print("Random Forest Results:\n")

print("Accuracy:", accuracy_score(y_test, rf_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

#Accuacy: 0.9907
#False Positives: 35
#False negatives: 28

#This is an extremely huge jump and insane good result
#Before analyzing further lets checking if there a data leakage or if we are accidentally including some identifier or label in the features

#print(X.columns) #This is just to check if we are using the correct features and not accidentally including some identifier or label in the features. The columns look fine and do not include any obvious identifiers or labels

#output showed no data leakage. We can begin with ROC + PR analysis to see if we can find a better operating point than 0.6 threshold of the baseline model.

rf_prob = rf_pipeline.predict_proba(X_test)      #for every test case it returns [P(benign), P(malware)]

# We must check whether probabilities are:
# [P(Benign), P(Malicious)] or reversed.

classes = rf_pipeline.named_steps["model"].classes_
print("Class order:", classes)

# Find index position of "Malicious"
mal_idx = list(classes).index("Malicious")

# Extract only the probability of being Malicious
rf_scores = rf_prob[:, mal_idx]

# for score in rf_scores:
#     print(score, end=", ")

from sklearn.metrics import roc_auc_score

# Convert y_test into binary (1 = Malicious, 0 = Benign)
y_test_binary = (y_test == "Malicious").astype(int)

# ROC-AUC measures ranking quality
rf_roc_auc = roc_auc_score(y_test_binary, rf_scores)

print("\nRF ROC-AUC:", rf_roc_auc)
#ROC - AUC: 0.99896
#This means if you randomly pick one malware file and one benign file, the model will correctly rank the malware higher 99.9% of the time.

from sklearn.metrics import average_precision_score

# PR-AUC focuses on precision vs recall tradeoff
rf_pr_auc = average_precision_score(y_test_binary, rf_scores)

print("RF PR-AUC:", rf_pr_auc)
#PR - AUC: 0.9983
#When the model tries to catch malware, it maintains extremely high precision and recall across almost all thresholds


#The results are excetionally good. The model is able to achieve very high accuracy, precision, recall, and AUC scores. 
# This EITHER suggests that Random Forest is able to capture complex patterns in the data that Logistic Regression could not
#Or that dataset is very seperable/has bias patterns/not clean or not adversarial enough

#Lets check for feature dominance
rf_model = rf_pipeline.named_steps["model"]

importances = rf_model.feature_importances_

feature_importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print(feature_importance_df.head(10))

#We observe that top features are mostly Compiler/OS/subsystem version metadata and not structural code features
#Hence there is dataset bias via build environment patterns

#In real world, malware makers can easily change these metadata patterns to evade detection
#Lets drop the Major and minor linker versions feature (not sturctural and top 2 features) and see if we can still achieve good performance
X_reduced = X.drop(columns=[
    "Minor_Linker_Version",
    "Major_Linker_Version"
])


#Repeating the process with reduced features
# ----------------------------------------------------------
# Retrain Random Forest WITHOUT linker version features
# ----------------------------------------------------------

X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(
    X_reduced,
    y,
    test_size=0.2,
    random_state=7,
    stratify=y
)

rf_pipeline_red = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])


rf_pipeline_red.fit(X_train_red, y_train_red)


rf_pred_red = rf_pipeline_red.predict(X_test_red)

print("\nRF WITHOUT Linker Features:\n")

print("Accuracy:", accuracy_score(y_test_red, rf_pred_red))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_red, rf_pred_red))
print("\nClassification Report:")
print(classification_report(y_test_red, rf_pred_red))



# Get probabilities
rf_prob_red = rf_pipeline_red.predict_proba(X_test_red)

# Extract malicious probability
classes_red = rf_pipeline_red.named_steps["model"].classes_
mal_idx_red = list(classes_red).index("Malicious")
rf_scores_red = rf_prob_red[:, mal_idx_red]

# Convert labels to binary
y_test_binary_red = (y_test_red == "Malicious").astype(int)

# Compute ROC-AUC
rf_roc_auc_red = roc_auc_score(y_test_binary_red, rf_scores_red)

# Compute PR-AUC
rf_pr_auc_red = average_precision_score(y_test_binary_red, rf_scores_red)

print("\nReduced RF ROC-AUC:", rf_roc_auc_red)
print("Reduced RF PR-AUC:", rf_pr_auc_red)

# Reduced RF ROC-AUC: 0.9971
# Reduced RF PR-AUC: 0.995
# Both values dropped by a little bit, which is a very good sign
#This indicates that linker version were contributing to RF's decision boundary a little bit
# But the model is still performing extremely well without them, which suggests that RF is able to capture other patterns well


# To get the features pertaining ONLY TO malware code structure and not build environment,
# We drop ALL VERSION features and only use the 

#Statement to add for research paper: Static malware detection models may over-rely on build-environment metadata rather than semantic malicious behavior.

#Lets put it to the test

# Remove ALL version-related metadata features
version_columns = [
    "Major_Image_Version",
    "Minor_Image_Version",
    "Major_Linker_Version",
    "Minor_Linker_Version",
    "Major_OS_Version",
    "Minor_OS_Version",
    "Major_Subsystem_Version",
    "Minor_Subsystem_Version"
]

X_no_versions = X.drop(columns=version_columns)

X_train_nv, X_test_nv, y_train_nv, y_test_nv = train_test_split(
    X_no_versions,
    y,
    test_size=0.2,
    random_state=7,
    stratify=y
)


rf_pipeline_nv = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

rf_pipeline_nv.fit(X_train_nv, y_train_nv)

rf_pred_nv = rf_pipeline_nv.predict(X_test_nv)

print("\nRF WITHOUT ANY VERSION FEATURES:\n")

print("Accuracy:", accuracy_score(y_test_nv, rf_pred_nv))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_nv, rf_pred_nv))
print("\nClassification Report:")
print(classification_report(y_test_nv, rf_pred_nv))



# Get probabilities
rf_prob_nv = rf_pipeline_nv.predict_proba(X_test_nv)

# Extract malicious probability
classes_nv = rf_pipeline_nv.named_steps["model"].classes_
mal_idx_nv = list(classes_nv).index("Malicious")
rf_scores_nv = rf_prob_nv[:, mal_idx_nv]

# Convert labels to binary
y_test_binary_nv = (y_test_nv == "Malicious").astype(int)

# Compute ROC-AUC
rf_roc_auc_nv = roc_auc_score(y_test_binary_nv, rf_scores_nv)

# Compute PR-AUC
rf_pr_auc_nv = average_precision_score(y_test_binary_nv, rf_scores_nv)

print("\nNo-Version RF ROC-AUC:", rf_roc_auc_nv)
print("No-Version RF PR-AUC:", rf_pr_auc_nv)

# Without ANY version metadata
# ROC-AUC ≈ 0.98205
# PR-AUC ≈ 0.97698
# Accuracy ≈ 0.922
# FN = 240
# FP = 290


# An attacker can:
    #Recompile malware using same toolchain as benign software
    #Modify header metadata
    #Spoof version fields

# If they do that:
    #Model performance may resemble the 0.92 case instead of 0.99 case.

#Lets perform a perturbation test:
# Instead of removing version features entirely:
    # Randomize version fields in test set only.
    # Simulate attacker spoofing metadata.
    # Then measure performance.

# If performance collapses → vulnerability confirmed.
# Else model may be learning more robust structural patterns beyond metadata.

