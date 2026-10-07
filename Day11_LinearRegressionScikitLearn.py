
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

# Select Features and Target Variable
# Target variable
y = df["Selling_Price"]

# Input features
X = df[
    [
        "Year",
        "Present_Price",
        "Kms_Driven",
        "Fuel_Type",
        "Seller_Type",
        "Transmission",
        "Owner"
    ]
]


# Encoding categorical variables
X = pd.get_dummies(
    X,
    columns=["Fuel_Type", "Seller_Type", "Transmission"],
    drop_first=True
)

print("\nFeatures after encoding:")
print(X.head())

print("\nFeature names:")
print(X.columns.tolist())



# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# 5. FEATURE SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression Model
linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train)
linear_pred = linear_model.predict(X_test_scaled)

# Print coefficients
print("\n" + "=" * 50)
print("LINEAR REGRESSION")
print("=" * 50)

print("\nIntercept:")
print(linear_model.intercept_)

print("\nCoefficients:")

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": linear_model.coef_
})

print(coefficients)


# Evaluation Function
def evaluate_model(model_name, y_actual, y_pred):

    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)

    print("\n" + model_name)
    print("-" * 30)
    print("MSE  :", mse)
    print("RMSE :", rmse)
    print("MAE  :", mae)
    print("R²   :", r2)

    return mse, rmse, mae, r2


# Evaluate Linear Regression
linear_results = evaluate_model(
    "Linear Regression",
    y_test,
    linear_pred
)

# Predicted vs Actual Plot
plt.figure(figsize=(8, 6))

plt.scatter(y_test, linear_pred)

# Reference diagonal line
minimum = min(y_test.min(), linear_pred.min())
maximum = max(y_test.max(), linear_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Linear Regression: Predicted vs Actual")

plt.tight_layout()
plt.savefig("predicted_vs_actual.png")

plt.show()


# Residual Plot
residuals = y_test - linear_pred
plt.figure(figsize=(8, 6))
plt.scatter(linear_pred, residuals)
plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Selling Price")
plt.ylabel("Residuals")
plt.title("Linear Regression: Residual Plot")
plt.tight_layout()
plt.savefig("residuals.png")
plt.show()

# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
ridge_pred = ridge_model.predict(X_test_scaled)
ridge_results = evaluate_model(
    "Ridge Regression",
    y_test,
    ridge_pred
)

# Lasso Regression
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(X_train_scaled, y_train)
lasso_pred = lasso_model.predict(X_test_scaled)
lasso_results = evaluate_model(
    "Lasso Regression",
    y_test,
    lasso_pred
)

