
import matplotlib.pyplot as p
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# **1. Data Generation**
x = np.linspace(0, 10, 100).reshape(-1, 1)  # Reshape for scikit-learn
noise = np.random.normal(0, 1, size=x.shape)  # Small random noise
y = 2 * x + 5 + noise  # Linear equation with noise

# **2. Model Training**
model = LinearRegression()
model.fit(x, y)

# **3. Model Evaluation**
coefficients = model.coef_[0][0]
intercept = model.intercept_[0]
r_squared = model.score(x, y)

print(f"Coefficient: {coefficients}")
print(f"Intercept: {intercept}")
print(f"R-Squared Score: {r_squared}")

# **4. Visualization**
y_pred = model.predict(x)  # Predict y values

p.scatter(x, y, label="Original Data", color="blue", alpha=0.5)
p.plot(x, y_pred, label="Regression Line", color="red")

# Add title, axis labels, and legend
p.title("Simple Linear Regression")
p.xlabel("X values")
p.ylabel("Y values")
p.legend()
p.show()
