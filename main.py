import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pandas import *

# Training Parameters
NORMALIZE_METHOD = "MINMAX"
TOTAL_NUM_INSTANCES = 10000
TRAIN_SPLIT = 0.8
LEARNING_RATE = 0.01
TRAINING_EPOCHS = 1000


# Normalizes features to Standard Normal Variable or maps them over [0,1] range
def featureNormalize(dataset):
    if(NORMALIZE_METHOD == "MINMAX"):
        print("Using min-max normalization")
        mins = np.amin(dataset, axis=0)
        maxs = np.amax(dataset, axis=0)
        return (dataset - mins) / (maxs-mins)
    else: 
        mu = np.mean(dataset,axis=0)
        sigma = np.std(dataset,axis=0)
        print("Using Z-Score Normalization with mean = ", mu, ", std dev = ", sigma)
        return (dataset-mu)/sigma

# Inserts a column of ones into the design matrix for Bias Terms
def prependBiasOnes(features):
  n_training_samples = features.shape[0]
  n_dim = features.shape[1]
  f = np.reshape(np.c_[np.ones(n_training_samples),features],[n_training_samples,n_dim + 1])
  return f

# Prints Matrices in a nicer way
def printM(dataset):
    print(DataFrame(dataset), "\n")


# Choose the fields to use as independant variables
# independentVars = ['sqft_living', 'sqft_lot', 'condition', 'waterfront', 'view', 'grade', 'sqft_above', 'sqft_basement']
independentVars = ['sqft_living', 'sqft_lot']
numFeatures = len(independentVars) # Doesnt include column of ones


# Read in the data
houseData = np.recfromcsv('houses_10k.csv', delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
houseData = houseData[:TOTAL_NUM_INSTANCES]
numInstances = len(houseData)


# Split our data
trainData = houseData[:round(numInstances*TRAIN_SPLIT)]
testData = houseData[round(numInstances*TRAIN_SPLIT):]
numTrainInstances = len(trainData)
numTestInstances = len(testData)
print("Training with ", numTrainInstances, " training instances")
print("Testing with ", numTestInstances, " testing instances")

# Create Design Matrices
xTest = np.zeros([numTestInstances, numFeatures])
xTrain = np.zeros([numTrainInstances, numFeatures])

# Fill Design Matrices
for index, var in enumerate(independentVars):
    xTrain[:,index] = trainData[var]
    xTest[:,index] = testData[var]

# Create output (y) vectors
yTrain = np.transpose(np.matrix(trainData['price']))
yTest = np.transpose(np.matrix(testData['price']))


# Normalize the features 
## NOTE: This may be incorrect as normalizes them with respect to test/train set
## Should probably normalize entire dataset first then split
xTrain = featureNormalize(xTrain)
xTest = featureNormalize(xTest)
yTrain = featureNormalize(yTrain)
yTest = featureNormalize(yTest)

# Add Bias Column to design matrices
xTrain = prependBiasOnes(xTrain)
xTest = prependBiasOnes(xTest)



###########################################
####         Begin Training         #######
###########################################
# Leaving as 'None' Rows so we can use for train and predict (different number of rows)
X = tf.placeholder(tf.float32, [None, numFeatures+1])   # Includes column of bias ones
Y = tf.placeholder(tf.float32,[None,1])
W = tf.Variable(tf.ones([numFeatures+1, 1]))
cost_history = np.empty(shape=[1],dtype=float)



# Specify Linear Regression method
init = tf.initialize_all_variables()
y_ = tf.matmul(X, W)
cost = tf.reduce_mean(tf.square(y_ - Y))
training_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

# Run the session
sess = tf.Session()
sess.run(init)

# Train
for epoch in range(TRAINING_EPOCHS):
    sess.run(training_step,feed_dict={X:xTrain,Y:yTrain})
    cost_history = np.append(cost_history, sess.run(cost,feed_dict={X: xTrain,Y: yTrain}))


###########################################
####         Evaluation             #######
###########################################
# Plot the cost function over time
plt.plot(range(len(cost_history)), cost_history, 'b+')
plt.xlabel("Epoch #")
plt.ylabel("Cost function")
plt.axis([0,TRAINING_EPOCHS,0,np.max(cost_history) + (0.1*np.max(cost_history))])
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

if(NORMALIZE_METHOD == "MINMAX"):
    plt.plot([0, 1, 2])
    plt.axis([0, 1.5, 0, 1.5])
else:
    plt.plot(range(-3,3), range(-3,3), 'b')
plt.xlabel("Predicted Price")
plt.ylabel("Actual Price")
plt.show()