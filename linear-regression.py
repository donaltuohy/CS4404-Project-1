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


# Performs gradient descent on training data
def trainModel(xTrain, yTrain, sess, trainingStep, lossFunctionHistory):
    for epoch in range(util.TRAINING_PARAMS['TRAINING_EPOCHS']):
        sess.run(trainingStep,feed_dict={X:xTrain,Y:yTrain})
        lossFunctionHistory = np.append(lossFunctionHistory, sess.run(cost,feed_dict={X: xTrain,Y: yTrain}))

    return lossFunctionHistory

# Returns the Mean Square and Mean Absolute Erros
def evaluateModel(yTest, predictedY):
    mse = sess.run(tf.reduce_mean(tf.square(predictedY - yTest)))
    mae = sess.run(tf.reduce_mean(tf.abs(predictedY - yTest)))

    return mse, mae


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
designMatrix = util.createDesignMatrix(data)
labelVector = util.createLabelVector(data)


designMatrix = util.featureNormalize(designMatrix)
labelVector = util.featureNormalize(labelVector)

designMatrix = prependBiasTerm(designMatrix)



###########################################
####     Model Definition           #######
###########################################
# Leaving as 'None' Rows so we can use for train and predict (different number of rows)
X = tf.placeholder(tf.float32, [None, designMatrix.shape[1]])
Y = tf.placeholder(tf.float32,[None,1])
W = tf.Variable(tf.ones([designMatrix.shape[1], 1]))
lossFunctionHistory = np.empty(shape=[1],dtype=float)

# Specify Linear Regression method and parameters
init = tf.global_variables_initializer()
yPredictor = tf.matmul(X, W)
cost = tf.reduce_mean(tf.square(Y-yPredictor))
trainingStep = tf.train.GradientDescentOptimizer(util.TRAINING_PARAMS['LEARNING_RATE']).minimize(cost)


# Create a session
sess = tf.Session()
sess.run(init)


# Train the model
if(util.TRAINING_PARAMS['SPLIT_METHOD'] == "KFOLD"):        
    mse = []
    mae = []
    numDataSplits = util.TRAINING_PARAMS['NUM_SPLITS']
    for i in range(numDataSplits):
        xTrain, yTrain, xTest, yTest = util.splitUpDataCrossVal(designMatrix, labelVector, numDataSplits, crossValIndex=i)
        trainModel(xTrain, yTrain, sess, trainingStep, lossFunctionHistory)
        predictedY = sess.run(yPredictor, feed_dict={X: xTest})
        currentMSE, currentMAE = evaluateModel(yTest, predictedY)
        mse.append(currentMSE)
        mae.append(currentMAE)
    print("MSE: ", mse)
    print("MAE: ", mae)
    averageMSE = np.mean(mse)
    averageMAE = np.mean(mae)

else:
    xTrain, yTrain, xTest, yTest = util.splitData7030(designMatrix, labelVector)
    trainModel(xTrain, yTrain, sess, trainingStep, lossFunctionHistory)
    predictedY = sess.run(yPredictor, feed_dict={X: xTest})
    averageMSE, averageMAE = evaluateModel(yTest, predictedY)

print("Most recent weights: ", np.matrix(sess.run(W)))
print("Average MSE: %4f" % averageMSE, ", Average MAE: %4f" % averageMAE)





###########################################
####         Evaluation             #######
###########################################
# # Plot the cost function over time
# plt.plot(range(len(lossFunctionHistory)), lossFunctionHistory, 'b+')
# plt.xlabel("Epoch #")
# plt.ylabel("Cost function")
# plt.axis([0,util.TRAINING_PARAMS['TRAINING_EPOCHS'],0,np.max(lossFunctionHistory) + (0.1*np.max(lossFunctionHistory))])
# plt.show()


# Plot first feature against predicter / measured values
plt.plot(xTest[:,1], yTest, 'ro')
plt.plot(xTest[:,1], predictedY, 'bx')
plt.xlabel("Feature 1")
plt.ylabel(util.ACTIVE_DATASET['LABEL'])
plt.show()

# # Plot Predicted y values against measured y values
plt.plot(predictedY, yTest, 'ro')

if((util.TRAINING_PARAMS['NORMALIZE_METHOD'] == "MINMAX")):
    # Plot over range from [0,1]
    plt.plot([0, 1])
else:
    # Plot over range [-3, 3] ~ 99 % of data
    plt.plot(range(-3,3), range(-3,3), 'b')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()