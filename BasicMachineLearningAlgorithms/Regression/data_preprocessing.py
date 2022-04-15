
#Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset

dataset=pd.read_csv("Data.csv")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

print(X)

print(Y)

# Taking care of missing data

from sklearn.impute import SimpleImputer
Imputer=SimpleImputer(missing_values=np.nan, strategy='median')
#simpleImputer used to handle missing data
Imputer.fit(X[:, 1:3])
X[:, 1:3]=Imputer.transform(X[:, 1:3])



print(X)

# Encoding categorical data

#Encoding the Independent Variable

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

print(X)

# Encoding the Dependent Variable

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
Y = label.fit_transform(Y)

print(Y)

#Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)

print(X_train)

print(X_test)

print(y_train)

print(y_test)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train[:,3:]=sc.fit_transform(X_train[:,3:])
X_test[:,3:]=sc.fit_transform(X_test[:,3:])

print(X_train)

print(X_test)
