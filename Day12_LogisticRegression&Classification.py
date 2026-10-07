import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Dataset Loading
df = pd.read_csv("Day12_Data.csv")
print("Dataset:")
print(df)
print("\nDataset Shape:")
print(df.shape)
print("\nFirst 5 rows:")
print(df.head())

# Define features and target

X = df[
    [
        "Hours_Studied",
        "Attendance",
        "Previous_Score",
        "Assignments_Completed"
    ]
]

y = df["Pass"]


# Plot the data

plt.figure(figsize=(8, 6))

plt.scatter(
    df["Hours_Studied"],
    df["Pass"]
)

plt.title("Hours Studied vs Pass/Fail")
plt.xlabel("Hours Studied")
plt.ylabel("Pass (0 = Fail, 1 = Pass)")

plt.show()


# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Initialize Logistic Regression model
model = LogisticRegression()


# Train the model
model.fit(X_train, y_train)


# Predict the test data
y_pred = model.predict(X_test)
print("\nActual Values:")
print(y_test.values)
print("\nPredicted Values:")
print(y_pred)


# Predict probability
y_probability = model.predict_proba(X_test)
print("\nPrediction Probabilities:")
print(y_probability)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:")
print(accuracy)
print("\nAccuracy Percentage:")
print(accuracy * 100, "%")


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

