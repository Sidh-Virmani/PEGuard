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

CSV_PATH = "dataset/PE_Dataset_Labeled.csv"

df = pd.read_csv(CSV_PATH)
 
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

print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))









