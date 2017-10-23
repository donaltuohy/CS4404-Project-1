####
####Loading in Data and organising it
####

#import util functions
from util import *

# Prepends a column of ones to design matrix to represent bias term 
def prependBiasTerm(dataset):
  numInstances = dataset.shape[0]
  numFeatures = dataset.shape[1]
  X = np.reshape(np.c_[np.ones(numInstances),dataset],[numInstances,numFeatures + 1])
  return X

#Use the util functions to read in the dataset and create required matrices
data = readData()
x = createDesignMatrix(data)
y = createLabelVector(data)

#Normalize the features
x = featureNormalize(x)
y = featureNormalize(y)

# Prepends a column of ones to design matrix to represent bias term 
x = prependBiasTerm(x)

#Split the data in the way specified in config.py
xTrain = splitData(x, y)[0]
yTrain = splitData(x, y)[1]
xTest = splitData(x, y)[2]
yTest = splitData(x,y)[3]

###
###Using sklearn to train
###


import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(xTrain, yTrain)

# Make predictions using the testing set
yPrediction = regr.predict(xTest)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean Squared Error: %.2f" % mean_squared_error(yTest, yPrediction))
print("Mean Absolute Error: %.2f" % mean_absolute_error(yTest, yPrediction))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(yTest, yPrediction))

# Plot outputs
#plt.plot(xTest[:,1], yTest, "ro" )
#plt.plot(xTest[:,1], yPrediction, "bo")
plt.plot(yTest, yPrediction, "ro")

plt. xlabel("Actual Value")
plt. ylabel("Predicted Value")
plt.xticks(())
plt.yticks(())

plt.show()