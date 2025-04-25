import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Boston housing dataset from OpenML
boston = fetch_openml(name='boston', version=1, as_frame=True)

# Convert to DataFrame
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['Price'] = boston.target

# Display sample rows
print("Sample data:")
print(data.head())

# Split features and target
X = data.drop('Price', axis=1)
y = data['Price']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to NumPy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error (MSE): {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Predict price for a new sample
print("\nEnter the values for each feature:")

features = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
    "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
]

new_input = []
for feature in features:
    value = float(input(f"{feature}: "))
    new_input.append(value)

new_input = np.array([new_input])
predicted_price = model.predict(new_input)

print(f"\nPredicted House Price: ${predicted_price[0]:.2f}")
