from util import (
    readData, 
    createDesignMatrix, 
    createLabelVector, 
    printM, 
    splitData7030, 
    splitUpDataCrossVal,
    featureNormalize
)
from config import TRAINING_PARAMS

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def plotFeatureAgainstClass(X, y):
    class0 = [x for i, x in enumerate(X) if (y[i] == 0)]
    class1 = [x for i, x in enumerate(X) if (y[i] == 1)]

    class0X, class0Y = np.asarray(class0).T
    class1X, class1Y = np.asarray(class1).T

    plt.plot(class0X, class0Y, 'r+')
    plt.plot(class1X, class1Y, 'bo')
    plt.show()


def evaluateTestPoints(sess, indicesOfKNN, trainPoints, trainClasses, testPoints, testClasses):
    sess.run(init)
    correctPredictions = 0

    # loop over test data points
    for i, testPoint in enumerate(testPoints):

        # Get nearest neighbors index
        nearestNeighboursIndices = sess.run(indicesOfKNN, feed_dict={currentTestPoint: testPoint, allTrainingPoints: trainPoints})

        # For categorical data (string labels) - only use first nearest neighbour
        if(TRAINING_PARAMS['IS_KNN_LABEL_STRING']):
            if(trainClasses[nearestNeighboursIndices[0]] == testClasses[i]):
                correctPredictions += 1
        else:
            # Compute the average value of the knn
            average = 0
            for index in nearestNeighboursIndices:
                average += trainClasses[index]
        
            average = average / TRAINING_PARAMS['K']

            # Defines how close is close enough for numeric classifications
            knnThreshold = TRAINING_PARAMS['KNN_CLASS_THRESHOLD']

            # Check if prediciton was correct
            if (knnThreshold):
                actual = testClasses[i]
                if(abs(actual - average) <= knnThreshold):
                    correctPredictions += 1
            elif average == testClasses[i]:
                correctPredictions += 1
    return correctPredictions / testPoints.shape[0]
        
data = readData()
points = createDesignMatrix(data)
classes = createLabelVector(data)
numFeatures = points.shape[1]



# tf Graph Input
allTrainingPoints = tf.placeholder("float", [None, numFeatures])
currentTestPoint = tf.placeholder("float", [numFeatures])


# Calculate L1 Distance between current test element and all of its neighbours
# Note taking negative so nn.top_k can find smallest
distance = tf.negative(tf.reduce_sum(tf.abs(tf.subtract(currentTestPoint, allTrainingPoints)), reduction_indices=1))

# Finds the top k nearest neighbours
values,indicesOfKNN=tf.nn.top_k(distance,k=TRAINING_PARAMS['K'],sorted=False)

correctPredictions = 0.
init = tf.global_variables_initializer()

# Evaluation
print("Starting Evaluation")
with tf.Session() as sess:
    if (TRAINING_PARAMS['SPLIT_METHOD'] == 'KFOLD'):
        accuracies = []
        numFolds = TRAINING_PARAMS['NUM_SPLITS']
        for i in range(numFolds):
            trainPoints, trainClasses, testPoints, testClasses = splitUpDataCrossVal(points, classes, numFolds, i)
            accuracy = evaluateTestPoints(sess, indicesOfKNN, trainPoints, trainClasses, testPoints, testClasses)
            print("Evaluating Fold ", i, " - Accuracy: ", accuracy)
            accuracies.append(accuracy)
    
        finalAccuracy = np.mean(accuracies)
    else:
        print("Evaluating using 70/30 split")
        trainPoints, trainClasses, testPoints, testClasses = splitData7030(points, classes)
        finalAccuracy = evaluateTestPoints(sess, indicesOfKNN, trainPoints, trainClasses, testPoints, testClasses)
    print("Evaluation Completed - Mean Accuracy: ", finalAccuracy)


