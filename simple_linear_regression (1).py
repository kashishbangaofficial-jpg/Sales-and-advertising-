# simple_linear_regression.py

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Load Dataset
# -----------------------------
# Make sure Data.xlsx is in the same folder
df = pd.read_excel("Data.xlsx")

print("Dataset Preview:")
print(df.head())

# -----------------------------
# Select Numeric Columns
# -----------------------------
numeric_cols = df.select_dtypes(include=['number']).columns

if len(numeric_cols) < 2:
    raise ValueError("Dataset must contain at least two numeric columns.")

# First numeric column as feature (X)
# Second numeric column as target (y)
X = df[[numeric_cols[0]]]
y = df[numeric_cols[1]]

print(f"\nFeature Column: {numeric_cols[0]}")
print(f"Target Column: {numeric_cols[1]}")

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Create and Train Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Model Evaluation
# -----------------------------
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# -----------------------------
# Model Coefficients
# -----------------------------
print("\nModel Details:")
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test, label="Actual Data")
plt.plot(X_test, y_pred, label="Regression Line")
plt.xlabel(numeric_cols[0])
plt.ylabel(numeric_cols[1])
plt.title("Simple Linear Regression")
plt.legend()
plt.show()
