from util import readData, createDesignMatrix, createLabelVector, printM, splitData, featureNormalize

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




data = readData()

points = createDesignMatrix(data)
classes = createLabelVector(data)

trainPoints, trainClasses, testPoints, testClasses = splitData(points, classes)
numFeatures = trainPoints.shape[1]
numTrainPoints = trainPoints.shape[0]
numTestPoints = testPoints.shape[0]




# tf Graph Input
allTrainingPoints = tf.placeholder("float", [None, numFeatures])
currentTestPoint = tf.placeholder("float", [numFeatures])


# Calculate L1 Distance between current test element and all of its neighbours
distance = tf.reduce_sum(tf.abs(tf.subtract(currentTestPoint, allTrainingPoints)), reduction_indices=1)
# Prediction: Predicted element is closest neighbour (second param is axis)
pred = tf.argmin(distance, 0) 


correctPredictions = 0.
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)

    # loop over test data points
    for i, testPoint in enumerate(testPoints):

        # Get nearest neighbor's index
        nearestNeighbourIndex = sess.run(pred, feed_dict={currentTestPoint: testPoint, allTrainingPoints: trainPoints})

        # Check if prediciton was correct
        if trainClasses[nearestNeighbourIndex] == testClasses[i]:
            correctPredictions += 1
        
        print("Test", i, "Predicted class:", trainClasses[nearestNeighbourIndex], "- True Class:", testClasses[i])

    print("Accuracy:", correctPredictions / numTestPoints)