
#Simple Linear Regression

#Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import datasets

db=pd.read_csv("Datasets/weight-height.csv")
x=db.iloc[:,1:-1].values
y=db.iloc[:,-1].values

print(x)

print(y)

#training datasets

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y=train_test_split(x,y, test_size=1/3,random_state=0)

print(train_x)

print(test_x)

print(train_y)

print(test_y)

#training the linear regression model

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(train_x,train_y)

#predicting test results
pred_y=regressor.predict(test_x)

#visualizing train set results
plt.scatter(train_x, train_y, color = 'red')
plt.plot(train_x, regressor.predict(train_x), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing test Results
plt.scatter(test_x, test_y, color = 'red')
plt.plot(test_x, regressor.predict(test_x), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
