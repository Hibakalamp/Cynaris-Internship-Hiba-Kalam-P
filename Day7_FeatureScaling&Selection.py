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

features = ["Age", "Annual_Income_INR", "Monthly_Spend_INR", "Electricity_Bill_INR"]
X = df[features]

# Normalization (Min-Max Scaling)
minmax_scaler = MinMaxScaler()
df_minmax = pd.DataFrame(minmax_scaler.fit_transform(X), columns=features)

print("MinMax Scaled Data:")
print(df_minmax.head())

# Standardization (Z-score Scaling)
standard_scaler = StandardScaler()
df_standard = pd.DataFrame(standard_scaler.fit_transform(X), columns=features)

print("\nStandardized Data:")
print(df_standard.head())