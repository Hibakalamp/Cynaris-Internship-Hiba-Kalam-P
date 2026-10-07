import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("Day9_Data.csv")

# Data inspection
print("Data Inspection")
print(df.info()) # display information about the DataFrame
print(df.describe()) # Display summary statistics
print(df.isnull().sum())  # Count of missing values in each column
print(df.shape) # display the shape of the DataFrame

# Define Features and Target
X = df["Review_Text"]
y = df["Sentiment"]

# Step 2: Encode Target Labels
le = LabelEncoder()
y = le.fit_transform(y)

# Convert Text to Numerical 
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression()
model.fit(X_train, y_train)
test_accuracy = model.score(X_test, y_test)
print("Train-Test Accuracy:", test_accuracy)

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
kf_scores = cross_val_score(model, X, y, cv=kf)
print("\nK-Fold Scores:", kf_scores)
print("K-Fold Mean Accuracy:", kf_scores.mean())

# Stratified K-Fold Cross Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf_scores = cross_val_score(model, X, y, cv=skf)
print("\nStratified K-Fold Scores:", skf_scores)
print("Stratified K-Fold Mean Accuracy:", skf_scores.mean())