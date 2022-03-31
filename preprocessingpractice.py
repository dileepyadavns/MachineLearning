# -*- coding: utf-8 -*-
"""preprocessingPractice

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yuCKOhS8lslU_8tEzhzi0JdZ2d2ngeWG

# Data Preprocessing Tools

## Importing the libraries
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""## Importing the dataset"""

dataset=pd.read_csv("Data.csv")
x=dataset.iloc[: ,:-1].values
y=dataset.iloc[:,-1].values

print(x)

print(y)

"""## Taking care of missing data"""

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

"""## Encoding categorical data"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np1.array(ct.fit_transform(x))

print(x)

"""### Encoding the Independent Variable"""



"""### Encoding the Dependent Variable

## Splitting the dataset into the Training set and Test set

## Feature Scaling
"""