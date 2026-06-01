import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("train.csv")

# Select input features
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]

# Select target variable (house price)
y = df['SalePrice']

# Create the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Predict the price of a new house
# Format: [Living Area, Bedrooms, Bathrooms]
prediction = model.predict([[2000, 3, 2]])

# Display the predicted price
print("Predicted House Price:", prediction[0])
