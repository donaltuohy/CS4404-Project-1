import util

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



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
        plt.ylabel(util.ACTIVE_DATASET['LABEL'])
        plt.show()



###########################################
####     Begin Data Organization    #######
###########################################

data = util.readData()
X = util.createDesignMatrix(data)
y = util.createLabelVector(data)


X = util.featureNormalize(X)
y = util.featureNormalize(y)

X = prependBiasTerm(X)

xTrain, yTrain, xTest, yTest = util.splitData7030(X, y)
# x10Train, y10Train, x10Test, y10Test = util.splitUpDataCrossVal(X, y, splitFactor=10)
util.splitUpDataCrossVal(X, y, splitFactor=10)

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
trainingStep = tf.train.GradientDescentOptimizer(util.TRAINING_PARAMS['LEARNING_RATE']).minimize(cost)

# Run the session
sess = tf.Session()
sess.run(init)

# Train
for epoch in range(util.TRAINING_PARAMS['TRAINING_EPOCHS']):
    sess.run(trainingStep,feed_dict={X:xTrain,Y:yTrain})
    lossFunctionHistory = np.append(lossFunctionHistory, sess.run(cost,feed_dict={X: xTrain,Y: yTrain}))


###########################################
####         Evaluation             #######
###########################################
# Plot the cost function over time
plt.plot(range(len(lossFunctionHistory)), lossFunctionHistory, 'b+')
plt.xlabel("Epoch #")
plt.ylabel("Cost function")
plt.axis([0,util.TRAINING_PARAMS['TRAINING_EPOCHS'],0,np.max(lossFunctionHistory) + (0.1*np.max(lossFunctionHistory))])
# plt.show()

# Predict Y Values for given test values
predictedY = sess.run(yPredictor, feed_dict={X: xTest})

# Calculate the Mean Square Error
mse = tf.reduce_mean(tf.square(predictedY - yTest))

# Calculate the Mean Absolute Error
mae = tf.reduce_mean(tf.abs(predictedY - yTest))

# Print Results
weights = np.matrix(sess.run(W))
print("Minimum Loss Function Value:", np.min(lossFunctionHistory),", MSE: %.4f" % sess.run(mse), ", MAE: %.4f" % sess.run(mae))
print("Weights: ", weights)


# Plot first feature against predicter / measured values
plt.plot(xTest[:,1], yTest, 'ro')
plt.plot(xTest[:,1], predictedY, 'bx')
plt.xlabel("Feature 1")
plt.ylabel(util.ACTIVE_DATASET['LABEL'])
# plt.show()

# Plot Predicted y values against measured y values
plt.plot(predictedY, yTest, 'ro')

if((util.TRAINING_PARAMS['NORMALIZE_METHOD'] == "MINMAX")):
    # Plot over range from [0,1]
    plt.plot([0, 1])
else:
    # Plot over range [-3, 3] ~ 99 % of data
    plt.plot(range(-3,3), range(-3,3), 'b')
plt.xlabel("Predicted")
plt.ylabel("Actual")
# plt.show()