import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np
import matplotlib.pyplot as plt

from numpy import genfromtxt

from sklearn.datasets import load_boston


def featureNormalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset)/sigma



# Read in the data
houseData = np.recfromcsv('houses_10k.csv', delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
houseData = houseData[:1000]
numInstances = len(houseData)


# Split our data
trainData = houseData[:round(numInstances/2)]
testData = houseData[round(numInstances/2):]
numTrainInstances = len(trainData)
numTestInstances = len(testData)


trainSqft = featureNormalize(trainData['sqft_living'])
trainPrices = featureNormalize(trainData['price'])

testSqft = featureNormalize(testData['sqft_living'])
testPrices = featureNormalize(testData['price'])

# Create Design Matrix X [[1...1][sqft1..sqftn]] as columns
trainX = np.column_stack((np.ones(numTrainInstances), trainSqft))
trainY = np.transpose(np.matrix(trainPrices))
numFeatures = trainX.shape[1]

# Create Test Data
testX = np.column_stack((np.ones(numTestInstances), trainSqft))
testY = np.transpose(np.matrix(testPrices))

print("trainX dimensions: ", trainX.shape, ", trainY Dimensions: ", trainY.shape)
print("testX dimensions: ", testX.shape, ", testY Dimensions: ", testY.shape)


learning_rate = 0.05
training_epochs = 20
cost_history = np.empty(shape=[1],dtype=float)


# Leaving as 'None' Rows so we can use for train and predict (different number of rows)
X = tf.placeholder(tf.float32, [None, numFeatures])
Y = tf.placeholder(tf.float32,[None,1])
W = tf.Variable(tf.ones([numFeatures, 1]))

# Set up for linear regression
init = tf.initialize_all_variables()

y_ = tf.matmul(X, W)
cost = tf.reduce_mean(tf.square(y_ - Y))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Run the session
sess = tf.Session()
sess.run(init)

# Train
for epoch in range(training_epochs):
    sess.run(training_step,feed_dict={X:trainX,Y:trainY})
    cost_history = np.append(cost_history, sess.run(cost,feed_dict={X: trainX,Y: trainY}))


# Plot the cost function over time
# plt.plot(range(len(cost_history)), cost_history, 'b+')
# plt.xlabel("Epoch #")
# plt.ylabel("Cost function")
# plt.axis([0,training_epochs,0,np.max(cost_history) + (0.1*np.max(cost_history))])
# plt.show()


pred_y = sess.run(y_, feed_dict={X: testX})
mse = tf.reduce_mean(tf.square(pred_y - testY))
weights = np.matrix(sess.run(W))
print("MSE: %.4f" % sess.run(mse))
print("Weights: ", weights)


plt.plot(testX[:,1], testY, 'ro')
plt.plot(testX[:,1], pred_y, 'bx')
plt.xlabel("Square Footage")
plt.ylabel("Price")
plt.show()