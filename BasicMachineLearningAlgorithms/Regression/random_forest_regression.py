
#Random forest Regression**

#Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#import dataset

df=pd.read_csv("Position_Salaries.csv")
x=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values

#training dataset for random forest regression

from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(x,y)

#predicting the result
print(regressor.predict([[6.5]]))

#Visualizing the result for Random Forest Regression(higher Resolution)
x_grid=np.arange(min(x),max(x),step=0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(x,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
plt.title("Truth or bluff (Random Forest Regression)")
plt.xlabel("Position")
plt.ylabel("Salaries")
plt.show()
