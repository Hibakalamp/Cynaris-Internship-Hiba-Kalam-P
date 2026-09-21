import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Day4_Dataset.csv")
print("Data Loaded Successfully")
print(df)

print("Data Inspection")
print(df.head())
print("\nSummary of Statistics")
print(df.describe())
print("\nInformation about the DataFrame")
df.info()
print("\nMissing Values Count")
print(df.isnull().sum())

# Documentation for the dataset
#1: The dataset contains 100 entries with 5 columns, including 4 numeric and 1 categorical column (Genre).
#2: There are no missing values in the dataset, meaning the data is clean and does not require imputation.
#3: The average age is ~39.75 years, with customers ranging from 18 to 70 years, showing a wide age distribution.
#4: The Annual Income ranges from 15k to 61k, with a mean of 39.56k, indicating moderate income variation among customers.
#5: The Spending Score ranges from 3 to 99, showing a strong variation in customer spending behavior, which is useful for segmentation analysis.


