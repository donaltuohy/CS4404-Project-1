#Assignment 1 for ML

#Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import util




data = util.readData()

# Prepends a column of ones to design matrix to represent bias term 
def prependBiasTerm(dataset):
  numInstances = dataset.shape[0]
  numFeatures = dataset.shape[1]
  X = np.reshape(np.c_[np.ones(numInstances),dataset],[numInstances,numFeatures + 1])
  return X
X = util.createDesignMatrix(data)
y = util.createLabelVector(data)


X = util.featureNormalize(X)
y = util.featureNormalize(y)

X = prependBiasTerm(X)

xTrain, yTrain, xTest, yTest = util.splitData(X, y)
"""
#Feature Scaling
from sklearn.preprocessing import StandardScaler
Scaler = StandardScaler()

Value_train = Scaler.fit_transform(Value_train)
Value_test = Scaler.transform(Value_test)


Area_train = Scaler.fit_transform(Area_train)
Area_test = Scaler.transform(Area_test)
"""

#Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xTrain, yTrain)

#reshape X to fit for Y
xTrain = xTrain[:,1]

#Preticting the Test set results
y_pred = regressor.predict(xTest)

#Mean square error
from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_pred,yTest)



#Visualising the Training Set results
plt.scatter(xTrain, yTrain, color = "red")
plt.plot(xTrain, regressor.predict(xTrain), color = "blue")
plt.title("Area vs Value (training Set)")
plt.xlabel("Area")
plt.ylabel("Value")

#Visualising the Test Set results
plt.scatter(xTest, yTest, color = "red")
plt.plot(xTrain, regressor.predict(xTest), color = "blue")
plt.title("Area vs Value (test Set)")
plt.xlabel("Area")
plt.ylabel("Value")