import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np
import matplotlib.pyplot as plt

from numpy import genfromtxt

from sklearn.datasets import load_boston


# Read in the data
houseData = np.recfromcsv('houses_10k.csv', delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
houseData = houseData[:100]
numInstances = len(houseData)


# Split our data
trainData = houseData[:round(numInstances/2)]
testData = houseData[round(numInstances/2):]
numTrainInstances = len(trainData)
numTestInstances = len(testData)


trainSqft = trainData['sqft_living']
trainPrices = trainData['price']

testSqft = testData['sqft_living']
testPrices = testData['price']

# Create Design Matrix X [[1...1][sqft1..sqftn]] as columns
trainX = np.column_stack((np.ones(numTrainInstances), trainSqft))
trainY = np.transpose(np.matrix(trainPrices))
numFeatures = trainX.shape[1]

# Create Test Data
testX = np.column_stack((np.ones(numTestInstances), trainSqft))
testY = np.transpose(np.matrix(testPrices))

print("trainX dimensions: ", trainX.shape, ", trainY Dimensions: ", trainY.shape)
print("testX dimensions: ", testX.shape, ", testY Dimensions: ", testY.shape)

# Plot the train data
# fig = plt.figure()
# plt.plot(trainSqft, trainPrices, 'ro')
# plt.xlabel('Sqft Living')
# plt.ylabel('Prices')
# plt.show()




learning_rate = 0.01
training_epochs = 1000
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
    # cost_history = np.append(cost_history, sess.run(cost,feed_dict={X: trainX,Y: trainY}))
    # print(sess.run(cost,feed_dict={X: trainX,Y: trainY}))

# Plot the cost function over time
plt.plot(range(len(cost_history)), cost_history)
plt.axis([0,training_epochs,0,20000000])
plt.show()


pred_y = sess.run(y_, feed_dict={X: testX})
mse = tf.reduce_mean(tf.square(pred_y - testY))
print("MSE: %.4f" % sess.run(mse))