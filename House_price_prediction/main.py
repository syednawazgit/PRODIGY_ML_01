# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the datasets
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
sample_submission = pd.read_csv('sample_submission.csv')

# Display first few rows of the training data
print("Training Data:")
print(train_data.head())

# Features and target variable
X = train_data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']]
y = train_data['SalePrice']

# Splitting the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on validation data
y_pred = model.predict(X_val)

# Evaluate the model
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE) on validation set: {rmse}")

# Predict on the test data
test_features = test_data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']]
test_predictions = model.predict(test_features)

# Create a submission file
submission = sample_submission.copy()
submission['SalePrice'] = test_predictions

# Save the submission file
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

# Optionally plot the results (predicted vs actual on validation set)
plt.figure(figsize=(10, 6)) 
plt.scatter(y_val, y_pred, color='blue', marker='o', label='Predicted Prices', alpha=0.6)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', linestyle='--', label='Ideal Prediction')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.legend() 
plt.grid(True)  
plt.show()