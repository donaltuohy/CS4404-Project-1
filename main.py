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
    csvData = np.recfromcsv(ACTIVE_DATASET['FILE_NAME'], delimiter=';', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
    numInstances = TRAINING_PARAMS['DESIRED_NUM_INSTANCES'] or len(csvData)
    csvData = csvData[:numInstances]
    return csvData


# Finds the column numbers of the features to be included in the design matrix X
def getDesiredFeatureIndices(allFeatureNames, ommitedFeatureNames):
    return [i for i, x in enumerate(allFeatureNames) if x not in ommitedFeatureNames]


# Builds the design matrix X inlcuding a column vector of 1s for the bias terms
def createDesignMatrix(dataset, omittedFeatures):
    numDesignMatrixFeatures = len(dataset[0]) - len(omittedFeatures) + 1    # include column of 1s for bias
    X = np.zeros((len(dataset), numDesignMatrixFeatures))

    featureIndices = getDesiredFeatureIndices(list(dataset.dtype.names), omittedFeatures)

    X[:,0] = 1
    currentCol = 1
    for i, row in enumerate(dataset):
        for j, col in enumerate(row):
            if(j in featureIndices):
                X[i,currentCol] = col
                currentCol = currentCol + 1
        currentCol = 1
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
    if(TRAINING_PARAMS['NORMALIZE_METHOD']):
        return scale(dataset)
    elif(NORMALIZE_METHOD == "MINMAX"):
        print("Using min-max normalization")
        mins = np.amin(dataset, axis=0)
        maxs = np.amax(dataset, axis=0)
        return (dataset - mins) / (maxs-mins)
    else: 
        mu = np.mean(dataset,axis=0)
        sigma = np.std(dataset,axis=0)
        print("Using Z-Score Normalization with mean = ", mu, ", std dev = ", sigma)
        return (dataset-mu)/sigma



data = readData()
X = createDesignMatrix(data, ACTIVE_DATASET['OMITTED_FEATURES'])
y = createLabelVector(data, ACTIVE_DATASET['LABEL'])

X = featureNormalize(X)
y = featureNormalize(y)
xTrain, yTrain, xTest, yTest = splitData(X, y)

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
cost_history = np.empty(shape=[1],dtype=float)



# Specify Linear Regression method
init = tf.global_variables_initializer()
y_ = tf.matmul(X, W)
cost = tf.reduce_mean(tf.square(Y-y_))
training_step = tf.train.GradientDescentOptimizer(TRAINING_PARAMS['LEARNING_RATE']).minimize(cost)

# Run the session
sess = tf.Session()
sess.run(init)

# Train
for epoch in range(TRAINING_PARAMS['TRAINING_EPOCHS']):
    sess.run(training_step,feed_dict={X:xTrain,Y:yTrain})
    cost_history = np.append(cost_history, sess.run(cost,feed_dict={X: xTrain,Y: yTrain}))


###########################################
####         Evaluation             #######
###########################################
# Plot the cost function over time
plt.plot(range(len(cost_history)), cost_history, 'b+')
plt.xlabel("Epoch #")
plt.ylabel("Cost function")
plt.axis([0,TRAINING_PARAMS['TRAINING_EPOCHS'],0,np.max(cost_history) + (0.1*np.max(cost_history))])
plt.show()

# Predict Y Values for given test values
pred_y = sess.run(y_, feed_dict={X: xTest})

# Calculate the Mean Square Error
mse = tf.reduce_mean(tf.square(pred_y - yTest))

# Print Results
weights = np.matrix(sess.run(W))
print("MSE: %.4f" % sess.run(mse))
print("Weights: ", weights)


## 1 Feature LR plot
# plt.plot(xTest[:,1], yTest, 'ro')
# plt.plot(xTest[:,1], pred_y, 'bx')
# plt.xlabel("Square Footage")
# plt.ylabel("Price")
# plt.show()

## Multi feature LR Plot
plt.plot(pred_y, yTest, 'ro')

if((TRAINING_PARAMS['NORMALIZE_METHOD'] == "MINMAX") or(TRAINING_PARAMS['NORMALIZE_METHOD'] == "SCALE")):
    plt.plot([0, 1, 2])
    plt.axis([0, 1.5, 0, 1.5])
else:
    plt.plot(range(-3,3), range(-3,3), 'b')
plt.xlabel("Predicted Price")
plt.ylabel("Actual Price")
plt.show()