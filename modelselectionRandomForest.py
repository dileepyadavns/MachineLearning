# -*- coding: utf-8 -*-
"""ms_RFC.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bm4-nn6awgvjCi5ZWVXza7vNVjlOK0km

**SVM classification**

**Importing the libraries**
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""**Importing the Data set**"""

df=pd.read_csv("Data.csv")
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

"""**Splitting the data set**"""

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.25,random_state=0)

"""**Feature Saclling**"""

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
train_x=sc.fit_transform(train_x)
test_x=sc.transform(test_x)

"""**Training the Data set for Decision Tree Classification**

"""

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(criterion='entropy',n_estimators=10,random_state = 0)

classifier.fit(train_x,train_y)

"""**Predicting test set results**"""

y_pred=classifier.predict(test_x)

"""**Confusion Matrix**"""

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_pred,test_y)
print(cm)
print(accuracy_score(y_pred,test_y))