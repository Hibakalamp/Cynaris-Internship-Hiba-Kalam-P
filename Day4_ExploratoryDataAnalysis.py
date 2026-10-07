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

# Displaying the distribution of numeric columns
num_cols = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]

for col in num_cols:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()

# Displaying heatmap of correlations between numeric columns
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True)
plt.show()

# Displaying the distribution of the categorical column "Genre"
plt.figure(figsize=(8, 6))
df["Genre"].value_counts().plot(kind="bar")
plt.show()

# EDA Narrative
# The dataset consists of 100 customer records with five columns: CustomerID, Genre, Age, Annual Income, and Spending Score.
# The dataset is clean and does not contain any missing values, so no data cleaning was required.
# The spending score varies from 3 to 99, which clearly shows that customer spending behavior is different among individuals.
# Some customers have low spending scores while others have high spending scores.

# It is also observed that there is no strong correlation between annual income and spending score. This means customers with
# higher income do not always spend more, and spending behavior depends onother factors as well.

# Overall, the dataset is well-structured and clean. It is suitable for analysis and visualization. 
# Further analysis like clustering can be used to group customers based on their behavior for better understanding and decision-making.