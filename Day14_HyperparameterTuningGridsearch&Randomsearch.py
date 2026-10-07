import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv("Day13_Data.csv")

# Remove ID column
data = data.drop("id", axis=1)

# Remove empty columns
data = data.dropna(axis=1, how="all")

# Convert diagnosis into numbers
data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

# Separate features and target
X = data.drop("diagnosis", axis=1)
y = data["diagnosis"]

# Convert features to numbers
X = X.apply(pd.to_numeric, errors="coerce")

# Fill missing values with median
X = X.fillna(X.median())

# Remove any remaining columns containing NaN
X = X.dropna(axis=1)

print("Features:", X.shape)
print("Missing values:", X.isnull().sum().sum())

# Target
y = data["diagnosis"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Scale data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ---------------- SVM ----------------

svm = SVC()

svm_params = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"]
}

svm_grid = GridSearchCV(
    svm,
    svm_params,
    cv=5,
    scoring="accuracy",
    error_score="raise"
)

print("Training data shape:", X_train.shape)
print("Target shape:", y_train.shape)
print("Missing values:", pd.isna(X_train).sum())
print("Starting SVM...")

svm_grid.fit(X_train, y_train)

print("\nSVM Best Parameters:")
print(svm_grid.best_params_)

svm_pred = svm_grid.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_pred)

print("SVM Accuracy:", svm_accuracy)


# ---------------- KNN ----------------

knn = KNeighborsClassifier()

knn_params = {
    "n_neighbors": range(1, 31),
    "weights": ["uniform", "distance"],
    "p": [1, 2]
}

knn_random = RandomizedSearchCV(
    knn,
    knn_params,
    n_iter=10,
    cv=5,
    random_state=42,
    scoring="accuracy",
    error_score="raise"
)

knn_random.fit(X_train, y_train)

print("\nKNN Best Parameters:")
print(knn_random.best_params_)

knn_pred = knn_random.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_pred)

print("KNN Accuracy:", knn_accuracy)


# ---------------- Comparison ----------------

print("\nModel Comparison")
print("SVM Accuracy:", svm_accuracy)
print("KNN Accuracy:", knn_accuracy)

if svm_accuracy > knn_accuracy:
    print("Best Model: SVM")
elif knn_accuracy > svm_accuracy:
    print("Best Model: KNN")
else:
    print("Both models have the same accuracy")