import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pandas import *
from sklearn.preprocessing import scale

# Load data from config file (see config.py)
from config import TRAINING_PARAMS, ACTIVE_DATASET




# Prints Matrices in a nicer way
def printM(dataset):
    print(DataFrame(dataset), "\n")

# Read's data from CSV File
def readData():
    csvData = np.recfromcsv(ACTIVE_DATASET['FILE_NAME'], delimiter=ACTIVE_DATASET['DELIMETER'], filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
    numInstances = TRAINING_PARAMS['DESIRED_NUM_INSTANCES'] or len(csvData)
    csvData = csvData[:numInstances]
    return csvData


# Finds the column numbers of the features to be included in the design matrix X
def getDesiredFeatureIndices(allFeatureNames, features, omitFeatures):
    return [i for i, x in enumerate(allFeatureNames) if ((x in features and not omitFeatures) or (x not in features and omitFeatures))]


# Builds the design matrix X inlcuding a column vector of 1s for the bias terms
def createDesignMatrix(dataset, features, omitFeatures):
    if(ACTIVE_DATASET['OMIT_FEATURES']):
        numDesignMatrixFeatures = len(dataset[0]) - len(features)
    else:
        numDesignMatrixFeatures = len(features)
    X = np.ones((len(dataset), numDesignMatrixFeatures))

    featureIndices = getDesiredFeatureIndices(list(dataset.dtype.names), features, omitFeatures)

    currentCol = 0
    for i, row in enumerate(dataset):
        for j, col in enumerate(row):
            if(j in featureIndices):
                X[i,currentCol] = col
                currentCol = currentCol + 1
        currentCol = 0
    return X


# Creates the specified label (output) vector from the dataset
def createLabelVector(dataset, labelName):
    y = np.transpose(np.matrix(dataset[labelName]))
    return y

# Splits data according to chosen split method
def splitData(X, y):
    if(TRAINING_PARAMS['SPLIT_METHOD'] == "70/30"):
        numTrainInstances = round(len(X)*0.7)
        xTrain = X[:numTrainInstances, :]
        yTrain = y[:numTrainInstances]
        xTest = X[numTrainInstances:]
        yTest = y[numTrainInstances:]
        return xTrain, yTrain, xTest, yTest



# Normalizes features to Standard Normal Variable or maps them over [0,1] range
def featureNormalize(dataset):
    if(TRAINING_PARAMS['NORMALIZE_METHOD'] == "MINMAX"):
        print("Using min-max normalization")
        mins = np.amin(dataset, axis=0)
        maxs = np.amax(dataset, axis=0)
        return (dataset - mins) / (maxs-mins)
    else: 
        mu = np.mean(dataset,axis=0)
        sigma = np.std(dataset,axis=0)
        print("Using Z-Score Normalization with mean = ", mu, ", std dev = ", sigma)
        return (dataset-mu)/sigma

# Prepends a column of ones to design matrix to represent bias term 
def prependBiasTerm(dataset):
  numInstances = dataset.shape[0]
  numFeatures = dataset.shape[1]
  X = np.reshape(np.c_[np.ones(numInstances),dataset],[numInstances,numFeatures + 1])
  return X

 # Plot some of the training data against input features
def showPlots(X, y):
    for i in range(1,X.shape[1]):
        plt.plot(X[:,i], y, 'r+')
        plt.xlabel("Feature %d" % i)
        plt.ylabel(ACTIVE_DATASET['LABEL'])
        plt.show()



###########################################
####     Begin Data Organization    #######
###########################################

data = readData()
X = createDesignMatrix(data, ACTIVE_DATASET['FEATURES'], ACTIVE_DATASET['OMIT_FEATURES'])
y = createLabelVector(data, ACTIVE_DATASET['LABEL'])


X = featureNormalize(X)
y = featureNormalize(y)

X = prependBiasTerm(X)

xTrain, yTrain, xTest, yTest = splitData(X, y)

# print(xTrain)
# showPlots(X,y)

numFeatures = len(xTrain[0])    # inclusive of bias column vector
numTrainInstances = len(yTrain)
numTestInstances = len(yTest)


###########################################
####         Begin Training         #######
###########################################
# Leaving as 'None' Rows so we can use for train and predict (different number of rows)
X = tf.placeholder(tf.float32, [None, numFeatures])
Y = tf.placeholder(tf.float32,[None,1])
W = tf.Variable(tf.ones([numFeatures, 1]))
lossFunctionHistory = np.empty(shape=[1],dtype=float)



# Specify Linear Regression method
init = tf.global_variables_initializer()
yPredictor = tf.matmul(X, W)
cost = tf.reduce_mean(tf.square(Y-yPredictor))
trainingStep = tf.train.GradientDescentOptimizer(TRAINING_PARAMS['LEARNING_RATE']).minimize(cost)

# Run the session
sess = tf.Session()
sess.run(init)

# Train
for epoch in range(TRAINING_PARAMS['TRAINING_EPOCHS']):
    sess.run(trainingStep,feed_dict={X:xTrain,Y:yTrain})
    lossFunctionHistory = np.append(lossFunctionHistory, sess.run(cost,feed_dict={X: xTrain,Y: yTrain}))


###########################################
####         Evaluation             #######
###########################################
# Plot the cost function over time
plt.plot(range(len(lossFunctionHistory)), lossFunctionHistory, 'b+')
plt.xlabel("Epoch #")
plt.ylabel("Cost function")
plt.axis([0,TRAINING_PARAMS['TRAINING_EPOCHS'],0,np.max(lossFunctionHistory) + (0.1*np.max(lossFunctionHistory))])
plt.show()

# Predict Y Values for given test values
predictedY = sess.run(yPredictor, feed_dict={X: xTest})

# Calculate the Mean Square Error
mse = tf.reduce_mean(tf.square(predictedY - yTest))

# Print Results
weights = np.matrix(sess.run(W))
print("Minimum Loss Function Value:", np.min(lossFunctionHistory),", MSE: %.4f" % sess.run(mse))
print("Weights: ", weights)


# Plot first feature against predicter / measured values
plt.plot(xTest[:,1], yTest, 'ro')
plt.plot(xTest[:,1], predictedY, 'bx')
plt.xlabel("Feature 1")
plt.ylabel(ACTIVE_DATASET['LABEL'])
plt.show()

# Plot Predicted y values against measured y values
plt.plot(predictedY, yTest, 'ro')

if((TRAINING_PARAMS['NORMALIZE_METHOD'] == "MINMAX")):
    # Plot over range from [0,1]
    plt.plot([0, 1])
else:
    # Plot over range [-3, 3] ~ 99 % of data
    plt.plot(range(-3,3), range(-3,3), 'b')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()