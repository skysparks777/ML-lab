import pandas as p
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as n


# 1
data = fetch_california_housing()
X = p.DataFrame(data.data, columns=data.feature_names)
Y = p.DataFrame()
Y['MedHouseVal'] = data.target
Y[data.feature_names] = X
print(Y.head())
print(Y.describe())

# 2
plt.hist(Y['MedHouseVal'], bins=30, edgecolor='black')
plt.xlabel('MedHouseVal')
plt.ylabel('yyyyy')
plt.show()

plt.scatter(Y['MedInc'], Y['MedHouseVal'], alpha=0.3)
plt.xlabel('MedInc')
plt.ylabel('MedHouseVal')
plt.show()

#3
a = Y[['MedInc']]
b = Y[['MedHouseVal']]

# 4
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size=0.2, random_state=42)


# 5
model = LinearRegression()
model.fit(a_train, b_train)

# 6
beta_0 = model.intercept_[0]
beta_1 = model.coef_[0][0]
print(f"Intercept (beta_0): {beta_0}")
print(f"Coefficient (beta_1): {beta_1}")


# 7
b_pred = model.predict(a_test)
mse = mean_squared_error(b_test, b_pred)
rmse = n.sqrt(mse)
r2 = r2_score(b_test, b_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared Score: {r2}")

# 8
plt.scatter(a_train, b_train)
plt.plot(a_train, model.predict(a_train))
plt.xlabel('MedInc')
plt.ylabel('MedHouseVal')
plt.legend()
plt.show()
