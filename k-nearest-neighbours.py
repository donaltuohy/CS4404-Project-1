from util import (
    readData, 
    createDesignMatrix, 
    createLabelVector, 
    printM, 
    splitData7030, 
    splitUpDataCrossVal,
    featureNormalize,
    multiclassToBinaryClass
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

    truePositives = 0
    falsePositives = 0
    trueNegatives = 0
    falseNegatives = 0

    # loop over test data points
    for i, testPoint in enumerate(testPoints):

        # Get nearest neighbors index
        nearestNeighboursIndices = sess.run(indicesOfKNN, feed_dict={currentTestPoint: testPoint, allTrainingPoints: trainPoints})

        # For categorical data (string labels) - only use first nearest neighbour
        if(TRAINING_PARAMS['IS_KNN_LABEL_STRING']):
            if(trainClasses[nearestNeighboursIndices[0]] == testClasses[i]):
                truePositives += 1
        else:
            # Compute the average value of the knn (just uses nearest neighbour if k=1)
            prediction = 0
            for index in nearestNeighboursIndices:
                prediction += trainClasses[index, 0]        # trainClasses is a list of lists (numpy)
            prediction = round(prediction / TRAINING_PARAMS['K'])

            # Defines how close is close enough for numeric classifications
            knnThreshold = TRAINING_PARAMS['KNN_CLASS_THRESHOLD']

            # Check if prediciton was correct
            if (knnThreshold):
                actual = testClasses[i]
                if(abs(actual - prediction) <= knnThreshold):
                    correctPredictions += 1
            else:
                correct = prediction == testClasses[i]
                if (correct and prediction == 1):
                    truePositives += 1
                elif (correct and prediction == 0):
                    trueNegatives += 1
                elif ((not correct) and prediction == 1):
                    falsePositives += 1
                else:
                    falseNegatives += 1
    return truePositives, falsePositives, trueNegatives, falseNegatives
        

def getMetrics(truePositives, falsePositives, trueNegatives, falseNegatives):
    accuracy = (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives)
    recall = (truePositives / (truePositives + falseNegatives)) 
    precision = truePositives / (truePositives + falsePositives)
    f1 = (recall * precision) / (precision + recall) 

    return accuracy, recall, precision, f1



data = readData()
points = createDesignMatrix(data)
classes = createLabelVector(data)
classes = multiclassToBinaryClass(classes, np.mean(classes))
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
        recalls = []
        precisions = []
        f1s = []
        numFolds = TRAINING_PARAMS['NUM_SPLITS']
        for i in range(numFolds):
            trainPoints, trainClasses, testPoints, testClasses = splitUpDataCrossVal(points, classes, numFolds, i)
            truePositives, falsePositives, trueNegatives, falseNegatives = evaluateTestPoints(sess, indicesOfKNN, trainPoints, trainClasses, testPoints, testClasses)
            
            print("Evaluating Fold ", i)
            confustionMatrix = np.matrix([[trueNegatives, falseNegatives], [falsePositives, truePositives]])
            print("Confusion matrix:")
            printM(confustionMatrix)

            accuracy, recall, precision, f1 = getMetrics(truePositives, falsePositives, trueNegatives, falseNegatives)
            accuracies.append(accuracy)
            recalls.append(recall)
            precisions.append(precision)
            f1s.append(f1)
            print("Accuracy: ", accuracy, ", Recall: ", recall, ", Precission: ", precision, ", F1: ", f1, "\n")
        
        accuracy = np.mean(accuracies)
        recall = np.mean(recalls)
        precision = np.mean(precisions)
        f1 = np.mean(f1)
    else:
        print("Evaluating using 70/30 split")
        trainPoints, trainClasses, testPoints, testClasses = splitData7030(points, classes)
        truePositives, falsePositives, trueNegatives, falseNegatives = evaluateTestPoints(sess, indicesOfKNN, trainPoints, trainClasses, testPoints, testClasses)
       
        confustionMatrix = np.matrix([[trueNegatives, falseNegatives], [falsePositives, truePositives]])
        print("Confusion matrix: ")
        printM(confustionMatrix)

        accuracy, recall, precision, f1 = getMetrics(truePositives, falsePositives, trueNegatives, falseNegatives)

    print("Evaluation Completed - Mean Accuracy: ", accuracy, ", Mean Recall: ", recall, ", Mean Precission: ", precision, ", meanF1: ", f1)


