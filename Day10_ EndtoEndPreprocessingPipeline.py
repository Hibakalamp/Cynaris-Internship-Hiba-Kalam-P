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
print()



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