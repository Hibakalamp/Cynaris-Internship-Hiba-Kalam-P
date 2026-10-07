
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("Day11_Data.csv")
print("Dataset loaded successfully!")
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset shape:")
print(df.shape)
print("\nColumn names:")
print(df.columns.tolist())