import pandas as pd
import numpy as np

# Train-test split
from sklearn.model_selection import train_test_split

# Pipeline
from sklearn.pipeline import Pipeline

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

# Feature selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# Machine Learning model
from sklearn.linear_model import LogisticRegression

# Evaluation
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

# Save and load pipeline
import joblib

# Load dataset
df = pd.read_csv("Day10_Data.csv")
print("Dataset loaded successfully!")

# Data inspection
print("First 5 rows:")
print(df.head())
print("\nDataset shape:")
print(df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nBasic statistics:")
print(df.describe())

# Select features and target variable
X = df.drop(["Student_ID", "Student_Name", "Passed"], axis=1)
# Passed is our target variable.
y = df["Passed"]
print("\nFeatures:")
print(X.columns.tolist())
print("\nTarget:")
print("Passed")

# Train-test split 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:")
print(X_train.shape)
print("\nTesting data shape:")
print(X_test.shape)

# Preprocessing pipeline
preprocessing_pipeline = Pipeline([
    
    # Handle missing values
    ("imputer", SimpleImputer(strategy="median")),
    
    # Standardize the features
    ("scaler", StandardScaler()),
    
    # Create polynomial features
    ("polynomial", PolynomialFeatures(
        degree=2,
        include_bias=False
    )),
    
    # Select the best features
    ("select_k_best", SelectKBest(
        score_func=f_classif,
        k=5
    ))
])


pipeline = Pipeline([
    
    ("preprocessing", preprocessing_pipeline),
    
    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])

# Display the pipeline
print("COMPLETE MACHINE LEARNING PIPELINE")
print(pipeline)

# Train the pipeline
print("\nTraining the pipeline...")
pipeline.fit(X_train, y_train)
print("Pipeline training completed!")

# 10. MAKE PREDICTIONS
y_pred = pipeline.predict(X_test)
print("\nPredictions:")
print(y_pred)

# Actual values
print("\nActual values:")
print(y_test.values)



# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("MODEL ACCURACY")
print(f"Accuracy: {accuracy * 100:.2f}%")


# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Save the pipeline
joblib.dump(
    pipeline,
    "student_performance_pipeline.pkl"
)

print("\nComplete pipeline saved successfully!")

# Load the pipeline
loaded_pipeline = joblib.load(
    "student_performance_pipeline.pkl"
)
print("Pipeline loaded successfully!")

# Predict using the loaded pipeline
loaded_predictions = loaded_pipeline.predict(X_test)
print("\nPredictions using loaded pipeline:")
print(loaded_predictions)

# Check the accuracy of the loaded pipeline
loaded_accuracy = accuracy_score(
    y_test,
    loaded_predictions
)

print("\nAccuracy of loaded pipeline:")
print(f"{loaded_accuracy * 100:.2f}%")

