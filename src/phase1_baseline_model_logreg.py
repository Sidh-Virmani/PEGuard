import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split     
from sklearn.preprocessing import StandardScaler        #for feature scaling
from sklearn.impute import SimpleImputer   #To replace NULL/NaN values with the median value
from sklearn.pipeline import Pipeline     #To ensure we preprocess + model together, in correct order
from sklearn.linear_model import LogisticRegression       #Our model lol
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#To know our accuracy
#Confusion matrix to know malware vs benign errors
#Classification report is for precision, F1, recall
from sklearn.model_selection import StratifiedKFold, cross_val_score
#To do cross validation to make sure our model is not overfitting 
from sklearn.model_selection import learning_curve      #For bias-variance diagnosis
from sklearn.model_selection import GridSearchCV   #For finding the best C for hyperparameter tuning
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
#For ROC-AUC and Precision-Recall AUC calculations to later compare models



CSV_PATH = "../dataset/PE_Dataset_Labeled.csv"

df = pd.read_csv(CSV_PATH)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
#Our data has first half as benign and second half as malware, shuffling it to ensure randomness
 
# print(df.shape)         #Viewing the data and checking whether file works
# print(df.columns)
# print(df.head())

#Target variable y is malware/benign present in the column named 'Label'
label_col = "Label"

#Looking at the dataset, the column "File_name" contains the name of the folder where file is present where its
#mentioned PE_files or Malware virus. Hence we remove it as well as the unnamed index column

#We therefore remove this identifier based leakage
df = df.drop(columns=["Unnamed: 0", "File_Name"])

#Getting our features and target variable
y = df["Label"]
X = df.drop(columns=["Label"])

#Lets start with only numeric features at first. We will introduce hex/string and other values overtime and observe
#marginal changes of our model based on them
X = X.select_dtypes(include=[np.number])       #only keeping numeric features

# print(X.shape)
# print(X.columns)
# print(X.head())
# print(y.shape)

#We dont use cross validation set for now since its a basic baseline model

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7, stratify=y
)
#freezing random state so every time we run we dont have a different test train split causing different accuracy
#we use the y (label column) as stratify so the ratio of malware to benign is almost the same in both test and train
#making it similar to entire dataset's ratio of malware to benign

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),   #Replacing NaN values with median of that column
    ("scaler", StandardScaler()),                    #Feature scaling so that all features are on same scale
    ("model", LogisticRegression(max_iter=1000))     #Logistic regression model, increasing max iterations to ensure convergence
])
#We have to do 3 things. Fill values -> Scale features -> Model training exactly in this order
#We have to do this everytime we use fit or predict. Hence we use Pipeline to automate this process
#For eg. while predicting on test data, we won't have to manually do all 3 we can simply use pipeline.predict()

pipeline.fit(X_train, y_train)     #does the 3 things in order

y_pred = pipeline.predict(X_test)   #also does prediction after the pipeline steps

print("------------------------------------------------------\n")
print("Baseline Model Results:\n")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("------------------------------------------------------\n")


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#StratifiedKFold to ensure each fold has similar ratio of malware to benign as entire dataset
#n_splits=5 splits data into 5 parts, each time using 1 part as test and rest 4 as train then rotates
#shuffle to ensure randomness
#freezing random state for reproducibility

cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

print("\nCross-validation accuracy scores:", cv_scores)
print("\nMean CV accuracy:", cv_scores.mean())
print("\nStd CV accuracy:", cv_scores.std())


#Results show our model is stable
#To make it even better we start with bias-variance diagnosis


train_sizes, train_scores, val_scores = learning_curve(
    pipeline,
    X,
    y,
    cv=cv,          #Table is sorted as in, almost first half is benign then second half is malware. So using cv = 5 will give error with 10% and 30% data, have to use stratiified k fold again to make they are in somewhat equal ratio
    scoring="accuracy",
    train_sizes=[0.1, 0.3, 0.5, 0.7, 1.0],   #Train data on 10%, 30%, 50%, 70% and 100% of data
    n_jobs=-1                    #Idk what this line does but internet says it makes the code run faster by making it use all CPU cores
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

print("------------------------------------------------------\n")

print("\nTraining accuracies:", train_mean)
print("\nValidation accuracies:", val_mean)
#Results show low bias and low variance, our model is good enough for baseline

print("------------------------------------------------------\n")

#C value in logisitic regression controls regularization
#Lower C means more regularization, higher C means less regularization
#If C too low then high bias, if C too high then high variance
#We will use grid search to find the best C value for our model

param_grid = {
    "model__C": [0.001, 0.01, 0.1, 1, 10, 100]       #works on logarithmic scale so 1.0 to 1.1 is too tiny
}

grid = GridSearchCV(                       #To find the best C value from above list
    pipeline,
    param_grid=param_grid,        
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\nBest C found:", grid.best_params_["model__C"])
print("\nBest cross-validation accuracy:", grid.best_score_)

# fine_grid = {                                       #finding more precise C value around the best C found
#     "model__C": [0.003, 0.005, 0.01, 0.02, 0.03]
# }

#Instead of manually finding more precise C values, we can use which automatically finds the best C for us
best_pipeline = grid.best_estimator_

#but now we have to use this pipeline for the model rather than the previous pipeline

#using best_pipeline for model fitting and prediction
best_pipeline.fit(X_train, y_train)     
y_pred = best_pipeline.predict(X_test)

print("------------------------------------------------------\n")
print("Baseline Model with Best C Results:\n")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


#Performing error analysis to see which files were misclassified
y_test_array = y_test.values

false_positives = np.where((y_test_array == "Benign") & (y_pred == "Malicious"))[0]

false_negatives = np.where((y_test_array == "Malicious") & (y_pred == "Benign"))[0]

print("\nError Analysis:")
print("\nFalse Positives (Benign → Malicious):", len(false_positives))
print("\nFalse Negatives (Malicious → Benign):", len(false_negatives))

print("------------------------------------------------------\n")


#We have done the following, very important and memorize for interviews/future projects

# Data loading + leakage removal
# Proper train/test split
# Pipeline-based preprocessing
# Baseline logistic regression
# Cross-validation
# Bias–variance check (learning curve)
# Hyperparameter tuning (C)
# Error analysis

#Result of error analysis showed 83 false positives and 207 false negatives
#But false negative is more dangerous since malware is classified as benign

#Lets try training another model (could have made simple changes in old but want to keep old records + new ones since
#this is for educational purposes) 
#In this model we will use the concept of cost weight to penalize false negatives more than false positives

cost_sensitive_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    (
        "model",
        LogisticRegression(
            max_iter=1000,
            class_weight={"Benign": 1, "Malicious": 2}           #Penalizing false negatives more by giving more weight to Malicious class
        )
    )
])

# Step 2: Train this new model
cost_sensitive_pipeline.fit(X_train, y_train)

# Step 3: Predict
y_pred_cs = cost_sensitive_pipeline.predict(X_test)

# Step 4: Evaluate
print("Cost-Sensitive Model Results:\n")

print("Accuracy:", accuracy_score(y_test, y_pred_cs))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_cs))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_cs))

# Step 5: Error analysis
y_test_array = y_test.values

false_positives_cs = np.where(
    (y_test_array == "Benign") & (y_pred_cs == "Malicious")
)[0]

false_negatives_cs = np.where(
    (y_test_array == "Malicious") & (y_pred_cs == "Benign")
)[0]

print("\nError Analysis (Cost-Sensitive):")
print("\nFalse Positives (Benign → Malicious):", len(false_positives_cs))
print("\nFalse Negatives (Malicious → Benign):", len(false_negatives_cs))

print("------------------------------------------------------\n")

#Result is 
#New false positives: 272
#New false negatives: 117

#Lets find a good optimal tradeoff between false positives and false negatives 
#We can try different class weights and see which gives the best tradeoff

weights_to_try = [1, 2, 3, 5]

for w in weights_to_try:
    print(f"\n--- Testing class_weight Malicious={w} ---")

    weighted_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=1000,
                class_weight={"Benign": 1, "Malicious": w}
            )
        )
    ])

    weighted_pipeline.fit(X_train, y_train)
    y_pred_w = weighted_pipeline.predict(X_test)

    cm = confusion_matrix(y_test, y_pred_w)
    fn = cm[1, 0]   # Malicious → Benign
    fp = cm[0, 1]   # Benign → Malicious

    print("False Negatives:", fn)
    print("False Positives:", fp)
    print("Malicious Recall:",
          classification_report(y_test, y_pred_w, output_dict=True)
          ["Malicious"]["recall"])
    
    
#Results:
# class_weight Malicious=1 
# False Negatives: 162
# False Positives: 151
# Malicious Recall: 0.9475048606610499

# class_weight Malicious=2
# False Negatives: 117
# False Positives: 272
# Malicious Recall: 0.9620868438107583


# class_weight Malicious=3
# False Negatives: 83
# False Positives: 423
# Malicious Recall: 0.9731043421905379

# class_weight Malicious=5
# False Negatives: 46
# False Positives: 445
# Malicious Recall: 0.9850939727802981

#Depending on use case we can choose the class weight accordingly
#For now, we will choose class_weight Malicious=3 as it gives a good tradeoff

print("------------------------------------------------------\n")

FINAL_MALICIOUS_WEIGHT = 3

final_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    (
        "model",
        LogisticRegression(
            max_iter=1000,
            C=grid.best_params_["model__C"],
            class_weight={"Benign": 1, "Malicious": FINAL_MALICIOUS_WEIGHT}
        )
    )
])

final_pipeline.fit(X_train, y_train)
final_pred = final_pipeline.predict(X_test)

print("\nFinal Model Results (Cost-Sensitive):")
print("\nAccuracy:", accuracy_score(y_test, final_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, final_pred))
print("\nClassification Report:")
print(classification_report(y_test, final_pred))

print("------------------------------------------------------\n")

#Final results:
#Accuracy dropped to 0.924 but false dropped from 207 to 83
#False positives increased from 272 to 423 but this is acceptable since false negatives are more dangerous
#Missing malware is worse than falsely flagging benign files

#ROC-AUC AND PR ANALYSIS:

#ROC is a measure of how fast the false alarms increase as we increase strictness of catching malware
#PR is a measure of how precise our model is when it comes to catching malware. Basically how often is the model right
#when it says a file is malware

y_prob = final_pipeline.predict_proba(X_test) 
#For each test sample, gives probability of being in each class in the following format:
#[probability_of_Benign , probability_of_Malicious]
#We dont know if its gonna be [P(benign), P(Malicious)] or [P(Malicious), P(Benign)] hence we have to find Malicious index lol

classes = final_pipeline.named_steps["model"].classes_
print("Class order:", classes)
malicious_index = list(classes).index("Malicious")
malware_scores = y_prob[:, malicious_index]
#Getting the probabilities of being Malicious for each test sample

fpr, tpr, roc_thresholds = roc_curve(
    y_test,
    malware_scores,
    pos_label="Malicious"
)

roc_auc = roc_auc_score(
    (y_test == "Malicious").astype(int),
    malware_scores
)

print("\nROC-AUC:", roc_auc)
#ROC full form is Receiver Operating Characteristic
#ROC-AUC means Area Under the Curve for ROC curve
#Higher the ROC-AUC better the model is (from 0 to 1)
#The value denotes "How much more is a malware file likely to be assigned a higher score of being malware than a benign file"
#0.5 means guessing, 0.8 means 80% better than guessing, 1.0 means perfect model

#Precision: Out of all files we flagged as malware, how many were actually malware?
#TP/(TP+FP)
#If 0.6 then 60% of all files flagged as malware were actually malware

#Recall: Out of all actual malware files, how many did we catch?
#TP/(TP+FN)
#If 0.95 then 95% of all malware files were caught

#It is only good that BOTH precision and recall are high, if one is high and other is low then there is some logical error, think

precision, recall, pr_thresholds = precision_recall_curve(
    (y_test == "Malicious").astype(int),
    malware_scores
)

pr_auc = average_precision_score(
    (y_test == "Malicious").astype(int),
    malware_scores
)

print("PR-AUC (Average Precision):", pr_auc)
#If PR-AUC is much lower than ROC-AUC, false positives explode when catching more malware

#The scores given by model needs to have a threshold to classify as malware or benign
#By default its 0.5, meaning if P(Malicious) > 0.5 then classify as Malicious else Benign
#But we can change this threshold to increase/decrease precision/recall as per our needs

#Lets try to find the best threshold
print("\nThreshold behavior:\n")

for thr in [0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1]:
    preds = np.where(malware_scores >= thr, "Malicious", "Benign")

    cm = confusion_matrix(
        y_test,
        preds,
        labels=["Benign", "Malicious"]
    )

    tn, fp, fn, tp = cm.ravel()

    precision_val = tp / (tp + fp) if (tp + fp) else 0
    recall_val = tp / (tp + fn) if (tp + fn) else 0
    fpr_val = fp / (fp + tn) if (fp + tn) else 0

    print(
        f"Threshold={thr} | "
        f"Precision={precision_val:.3f} | "
        f"Recall={recall_val:.3f} | "
        f"FPR={fpr_val:.3f} | "
        f"FP={fp} FN={fn}"
    )


#Final results show that at threshold 0.6 and 0.5 we get a good balance of precision and recall
#Threshold=0.6 | Precision=0.921 | Recall=0.964 | FPR=0.068 | FP=255 FN=110
#Threshold=0.5 | Precision=0.875 | Recall=0.971 | FPR=0.115 | FP=429 FN=88

#Choosing threshold=0.6 for final model as it gives a good balance since at 0.5 FP explodes

#Can any other Model of feature engineering + data manipulation beat this baseline?

#Next goal: Can Random Forest achieve recall ≥ 0.964 with FP < 255?



#End of baseline model training