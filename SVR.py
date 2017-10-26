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
sc_xTrain = StandardScaler()
sc_yTrain = StandardScaler()
sc_xTest = StandardScaler()
sc_yTest = StandardScaler()
xTrain = sc_xTrain.fit_transform(xTrain)
sc_xTest = sc_xTest.fit_transform(xTest)
sc_yTrain = sc_yTrain.fit_transform(yTrain)
sc_yTest = sc_yTest.fit_transform(yTest)
"""
#Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(xTrain,yTrain)

#Preticting the Test set results
y_pred = regressor.predict(xTest)

#Mean square error
from sklearn.metrics import mean_squared_error



rmse = mean_squared_error(y_pred,yTest)

print(rmse)
#reshape X to fit for Y
"""xTrain = xTrain[:,1]

#Visualising the Training Set results
plt.scatter(xTrain, yTrain, color = "red")
plt.plot(xTrain, regressor.predict(xTrain), color = "blue")
plt.title("Area vs Value (training Set)")
plt.xlabel("Area")
plt.ylabel("Value")"""


