import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Day4_Dataset.csv")

print("Dataset Loaded Successfully")
print(df.head())

print("\nDataFrame Information:")
print(df.info())
print(df.describe())

# Displaying the distribution of Age using a histogram 
plt.figure(figsize=(6,4))
plt.hist(df["Age"], bins=10, color="Red", edgecolor="black")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Displaying the distribution of Spending Score using a Boxplot
plt.figure(figsize=(6,4))
sns.boxplot(y=df["Spending Score (1-100)"], color="orange")
plt.title("Spending Score Outliers")
plt.show()

#  Displaying the distribution of Income vs Spending Score using a scatter plot
plt.figure(figsize=(6,4))
plt.scatter(df["Annual Income (k$)"], df["Spending Score (1-100)"], color="green")
plt.title("Income vs Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.show()

# Displaying the distribution of Gender using a bar plot 
plt.figure(figsize=(6,4))
df["Genre"].value_counts().plot(kind="bar", color=["blue", "pink"])
plt.title("Gender Count")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.show()

# Displaying the trend of Age vs Index using a line plot
plt.figure(figsize=(6,4))
plt.plot(df["Age"], color="purple")
plt.title("Age Trend")
plt.xlabel("Index")
plt.ylabel("Age")
plt.show()

# Displaying the correlation heatmap of the dataset
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()