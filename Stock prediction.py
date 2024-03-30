import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('C:/Users/lenovo/Desktop/Apple.csv')


# Selecting the features (input) and target (output) variables
X = data[['Open', 'High', 'Low', 'Volume']]  # Features
y = data['Close']  # Target variable

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", np.sqrt(mse))

# Example prediction for a new data point
new_data_point = [[123.45, 125.67, 122.34, 1000000]]  # Example data point with Open, High, Low, Volume
predicted_close_price = model.predict(new_data_point)
print("Predicted Close Price:", predicted_close_price)

# Plotting actual vs. predicted closing prices
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Close'], label='Actual Close Price', color='blue')
plt.plot(data.index, model.predict(X), label='Predicted Close Price', color='red')
plt.xlabel('Index')
plt.ylabel('Closing Price')
plt.title('Actual vs. Predicted Closing Prices')
plt.legend()
plt.tight_layout()
plt.show()
