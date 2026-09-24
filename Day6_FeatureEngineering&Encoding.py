import pandas as pd

df = pd.read_csv("Day4_Dataset.csv")
print(df.head())

# Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Genre_encoded'] = le.fit_transform(df['Genre'])

print(df[['Genre', 'Genre_encoded']].head())

# One-Hot Encoding
df1 = pd.get_dummies(df, columns=['Genre'], prefix='Genre')
print(df1.head())


num_cols = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = df[num_cols]
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Standard Scaling
scaler_std = StandardScaler()
df_std = scaler_std.fit_transform(X)

# Min-Max Scaling
scaler_minmax = MinMaxScaler()
df_minmax = scaler_minmax.fit_transform(X)

# Robust Scaling
scaler_robust = RobustScaler()
df_robust = scaler_robust.fit_transform(X) 

df_std = StandardScaler().fit_transform(X)
df_minmax = MinMaxScaler().fit_transform(X)

# Visualization (Before vs After)
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Before scaling
axes[0].hist(X['Age'], bins=10)
axes[0].set_title("Original Age Distribution")

# Standard Scaled
axes[1].hist(df_std[:, 0], bins=10)
axes[1].set_title("Standard Scaled Age")

# MinMax Scaled
axes[2].hist(df_minmax[:, 0], bins=10)
axes[2].set_title("MinMax Scaled Age")

plt.tight_layout()
plt.show()

#Feature Selection (SelectKBest)
from sklearn.feature_selection import SelectKBest, f_classif

X_features = df[['Age', 'Annual Income (k$)', 'Genre_encoded']]
y = df['Spending Score (1-100)']

selector = SelectKBest(score_func=f_classif, k=3)
fit = selector.fit(X_features, y)

scores = pd.DataFrame({
    'Feature': X_features.columns,
    'Score': fit.scores_
})

print(scores.sort_values(by='Score', ascending=False))