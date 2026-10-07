import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load Dataset
df = pd.read_csv("Day13_Data.csv")
print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(df.head())

# Data Inspection
print("\nDataset Shape:")
print(df.shape)
print("\nDataset Information:")
df.info()
print("\nColumn Names:")
print(df.columns.tolist())
print("\nStatistical Summary:")
print(df.describe())

# 3. Check Number of Classes
print("\nNumber of Classes:")
print(df["diagnosis"].nunique())
print("\nClasses:")
print(df["diagnosis"].unique())
print("\nClass Distribution:")
print(df["diagnosis"].value_counts())


# Check for Missing Values
print("\nMissing Values in Each Column:")
print(df.isnull().sum())
print("\nTotal Missing Values:")
print(df.isnull().sum().sum())


# Data Visualization- Scatter Plot and Pair Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df,
    x="radius_mean",
    y="texture_mean",
    hue="diagnosis"
)
plt.title("Radius Mean vs Texture Mean")
plt.xlabel("Radius Mean")
plt.ylabel("Texture Mean")
plt.show()

pairplot_features = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "diagnosis"
]
sns.pairplot(
    df[pairplot_features],
    hue="diagnosis"
)
plt.show()

# Prepare Features and Target
# Drop ID because it is only an identifier
X = df.drop(columns=["id", "diagnosis"])

# Target variable
y = df["diagnosis"]

# Convert diagnosis:
# M = 1
# B = 0

y = y.map({
    "M": 1,
    "B": 0
})

print("\nFeatures Shape:")
print(X.shape)
print("\nTarget Shape:")
print(y.shape)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("\nTraining Data Shape:")
print(X_train.shape)
print("\nTesting Data Shape:")
print(X_test.shape)

# Train Decision Tree Model
decision_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

decision_tree.fit(X_train, y_train)
print("\nDecision Tree model trained successfully!")


# Visualize Decision Tree
plt.figure(figsize=(20, 12))
plot_tree(
    decision_tree,
    feature_names=X.columns,
    class_names=["Benign", "Malignant"],
    filled=True,
    rounded=True,
    max_depth=3
)

plt.title("Decision Tree Visualization")
plt.show()


# Decision Tree Predictions
y_pred_tree = decision_tree.predict(X_test)
print("\nDecision Tree Predictions:")
print(y_pred_tree)


# Decision Tree Evaluation
tree_accuracy = accuracy_score(y_test, y_pred_tree)
print("\nDecision Tree Accuracy:")
print(f"{tree_accuracy * 100:.2f}%")
print("\nDecision Tree Classification Report:")
print(
    classification_report(
        y_test,
        y_pred_tree,
        target_names=["Benign", "Malignant"]
    )
)



# Decision Tree Confusion Matrix
tree_cm = confusion_matrix(y_test, y_pred_tree)
print("\nDecision Tree Confusion Matrix:")
print(tree_cm)

# Decision Tree Confusion Matrix Heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(
    tree_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"]
)

plt.title("Decision Tree - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Train Random Forest Model
random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest.fit(X_train, y_train)
print("\nRandom Forest model trained successfully!")



# Random Forest Predictions
y_pred_forest = random_forest.predict(X_test)
print("\nRandom Forest Predictions:")
print(y_pred_forest)

# Random Forest Evaluation
forest_accuracy = accuracy_score(y_test, y_pred_forest)
print("\nRandom Forest Accuracy:")
print(f"{forest_accuracy * 100:.2f}%")

print("\nRandom Forest Classification Report:")
print(
    classification_report(
        y_test,
        y_pred_forest,
        target_names=["Benign", "Malignant"]
    )
)

# Random Forest Confusion Matrix
forest_cm = confusion_matrix(y_test, y_pred_forest)
print("\nRandom Forest Confusion Matrix:")
print(forest_cm)


# Random Forest Confusion Matrix Heatmap
plt.figure(figsize=(7, 5))
sns.heatmap(
    forest_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Benign", "Malignant"],
    yticklabels=["Benign", "Malignant"]
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Compare Models
print("\n MODEL COMPARISON ")

print(f"Decision Tree Accuracy : {tree_accuracy * 100:.2f}%")
print(f"Random Forest Accuracy : {forest_accuracy * 100:.2f}%")