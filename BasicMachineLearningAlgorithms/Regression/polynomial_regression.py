

# Polynomial Regression

# Importing the libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset

df=pd.read_csv("Datasets/Position_Salaries.csv")
x = df.iloc[:, 1:-1].values
y = df.iloc[:, -1].values

# Training the Linear Regression model on the whole dataset

from sklearn.linear_model import LinearRegression
le=LinearRegression()
le.fit(x,y)

# Training the Polynomial Regression model on the whole dataset

from sklearn.preprocessing import PolynomialFeatures
polyReg=PolynomialFeatures(degree=4)
poly_x=polyReg.fit_transform(x)

linear_reg1=LinearRegression()
linear_reg1.fit(poly_x,y)

# Visualising the Linear Regression results
plt.scatter(x,y,color='red')
plt.plot(x,le.predict(x),color='blue')
plt.title("Linear REg")
plt.xlabel("position")
plt.ylabel("Salaries")
plt.show()

# Visualising the Polynomial Regression results

plt.scatter(x,y,color='red')
plt.plot(x, linear_reg1.predict(polyReg.fit_transform(x)),color='blue')
plt.title("Linear REg")
plt.xlabel("position")
plt.ylabel("Salaries")
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(x), max(x), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(x, y, color = 'red')
plt.plot(X_grid, linear_reg1.predict(polyReg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression

y_pred=le.predict([[6.5]])
print(y_pred)

# Predicting a new result with Polynomial Regression
y_pred1=linear_reg1.predict(polyReg.fit_transform([[6.5]]))
print(y_pred1)
