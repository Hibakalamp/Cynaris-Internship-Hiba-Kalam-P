import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load dataset
df = pd.read_csv("Day7_Data.csv")

# Data inspection
print("Data Inspection")
print(df.info()) # display information about the DataFrame
print(df.describe()) # Display summary statistics
print(df.isnull().sum())  # Count of missing values in each column
print(df.shape) # display the shape of the DataFrame

