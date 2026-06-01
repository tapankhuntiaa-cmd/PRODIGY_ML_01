import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("train.csv")

X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['SalePrice']

model = LinearRegression()

model.fit(X, y)

prediction = model.predict([[2000, 3, 2]])

print(prediction)