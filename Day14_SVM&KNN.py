import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# Load Dataset
df = pd.read_csv("Day13_Data.csv")
print("Dataset Loaded Successfully!")
print()


# Data Inspection
print("First 5 Rows:")
print(df.head())
print()
print("Dataset Shape:")
print(df.shape)
print()
print("Column Names:")
print(df.columns)
print()
print("Dataset Information:")
df.info()
print()
print("Statistical Summary:")
print(df.describe())
print()
print("Missing Values:")
print(df.isnull().sum())
print()
print("Target Class Distribution:")
print(df["diagnosis"].value_counts())


# Data Preparation
# Remove unnecessary ID column
df = df.drop("id", axis=1)

# Convert diagnosis:
# M = 1 (Malignant)
# B = 0 (Benign)
df["diagnosis"] = df["diagnosis"].map({
    "M": 1,
    "B": 0
})

# Separate features and target
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# Handle infinite and missing values
X = X.replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how="all")
X = X.fillna(X.median())

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\nFeature scaling completed.")


# KNN Model
knn = KNeighborsClassifier(n_neighbors=5)

# Train KNN
knn.fit(X_train_scaled, y_train)

# Make predictions
knn_predictions = knn.predict(X_test_scaled)

# Calculate accuracy
knn_accuracy = accuracy_score(y_test, knn_predictions)

print("\n================ KNN RESULTS ================")
print("KNN Accuracy:", knn_accuracy)

# Confusion Matrix
knn_cm = confusion_matrix(y_test, knn_predictions)

print("\nKNN Confusion Matrix:")
print(knn_cm)

# Classification Report
print("\nKNN Classification Report:")
print(classification_report(y_test, knn_predictions))


# SVM Model
svm = SVC(
    kernel="rbf",
    C=1.0
)

# Train SVM
svm.fit(X_train_scaled, y_train)

# Make predictions
svm_predictions = svm.predict(X_test_scaled)

# Calculate accuracy
svm_accuracy = accuracy_score(y_test, svm_predictions)

print("\n================ SVM RESULTS ================")
print("SVM Accuracy:", svm_accuracy)

# Confusion Matrix
svm_cm = confusion_matrix(y_test, svm_predictions)

print("\nSVM Confusion Matrix:")
print(svm_cm)

# Classification Report
print("\nSVM Classification Report:")
print(classification_report(y_test, svm_predictions))


# Compare KNN and SVM
comparison = pd.DataFrame({
    "Model": ["KNN", "SVM"],
    "Accuracy": [knn_accuracy, svm_accuracy]
})
print("\n================ MODEL COMPARISON ================")
print(comparison)


# Visualize Accuracy Comparison
plt.figure(figsize=(7, 5))
plt.bar(
    comparison["Model"],
    comparison["Accuracy"]
)

plt.ylim(0, 1.1)
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("KNN vs SVM Accuracy Comparison")
plt.show()


# Display Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# KNN Confusion Matrix
axes[0].imshow(knn_cm)
axes[0].set_title("KNN Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")

for i in range(knn_cm.shape[0]):
    for j in range(knn_cm.shape[1]):
        axes[0].text(
            j,
            i,
            knn_cm[i, j],
            ha="center",
            va="center"
        )


# SVM Confusion Matrix
axes[1].imshow(svm_cm)
axes[1].set_title("SVM Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

for i in range(svm_cm.shape[0]):
    for j in range(svm_cm.shape[1]):
        axes[1].text(
            j,
            i,
            svm_cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.show()