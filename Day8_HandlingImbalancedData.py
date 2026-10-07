import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

df = pd.read_csv("Day8_Data.csv")

# Data inspection
print("Data Inspection")
print(df.head())
print(df.info()) # display information about the DataFrame
print(df.describe()) # Display summary statistics
print(df.isnull().sum())  # Count of missing values in each column
print(df.shape) # display the shape of the DataFrame


# Splitting the dataset into features and target variable
X = df.drop("Default", axis=1)
y = df["Default"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Baseline Model (Logistic Regression)
baseline = LogisticRegression(max_iter=1000)
baseline.fit(X_train, y_train)
pred_base = baseline.predict(X_test)

print("\nBASELINE MODEL")
print(classification_report(y_test, pred_base))



# Smote (OVERSAMPLING)
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
model_smote = LogisticRegression(max_iter=1000)
model_smote.fit(X_smote, y_smote)
pred_smote = model_smote.predict(X_test)

print("\nAFTER SMOTE")
print(classification_report(y_test, pred_smote))



# Tomek Links (UNDERSAMPLING)
tomek = TomekLinks()
X_tomek, y_tomek = tomek.fit_resample(X_train, y_train)
model_tomek = LogisticRegression(max_iter=1000)
model_tomek.fit(X_tomek, y_tomek)
pred_tomek = model_tomek.predict(X_test)

print("\nAFTER TOMEK LINKS")
print(classification_report(y_test, pred_tomek))