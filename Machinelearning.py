import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
data = pd.read_csv(r'C:\Users\Admin\OneDrive\Desktop\python\placement_data.csv')

# Rename columns for consistency
data.rename(columns={'Package (LPA)': 'Placement_Package', 'Cource': 'Course'}, inplace=True)

# One-Hot Encoding for the "Course" column
data = pd.get_dummies(data, columns=["Course"], drop_first=True)

# Define Features (X) and Target (y)
X = data[['CGPA'] + [col for col in data.columns if col.startswith('Course_')]]
y = data['Placement_Package']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Calculate Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print Results
print("Mean Squared Error:", mse)
print("R-Squared Value:", r2)

# Visualization
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel("Actual Packages")
plt.ylabel("Predicted Packages")
plt.title("Actual vs Predicted Packages")
plt.show()

# Model Details
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)