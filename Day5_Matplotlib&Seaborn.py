import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Day4_Dataset.csv")

print("Dataset Loaded Successfully")
print(df.head())

print("\nDataFrame Information:")
print(df.info())
print(df.describe())