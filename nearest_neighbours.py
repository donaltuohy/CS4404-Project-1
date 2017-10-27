from util import *
import pandas as pd
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors

def trainModel(xtrain, ytrain, xtest, ytest):
    clf = neighbors.KNeighborsClassifier(TRAINING_PARAMS['NUMBER_NEIGHBORS'])
    clf.fit(xtrain, ytrain)
    yPrediction = clf.predict(xtest)
    accuracy = clf.score(xtest,ytest)
    return accuracy, yPrediction

def calculatePrecision(xtest, ytest, yPrediction):
    truePositives = 0
    falsePositives = 0
    trueNegatives = 0
    falseNegatives = 0

    for i in range(len(xtest)):
        if (yPrediction[i] == 0) and (ytest[i] == 0):
            trueNegatives += 1
        elif (yPrediction[i] == 0) and (ytest[i] == 1):
            falseNegatives += 1
        elif (yPrediction[i] == 1) and (ytest[i] == 1):
            truePositives += 1
        elif (yPrediction[i] == 1) and (ytest[i] == 0):
            falsePositives += 1

    #Return Precision
    return truePositives/ (truePositives + falsePositives)
    

data = readData()
x = createDesignMatrix(data)
y = createLabelVector(data)
y = multiclassToBinary(y, np.mean(y))

if(TRAINING_PARAMS['SPLIT_METHOD'] == "KFOLD"):        
    acc = []
    pre = []
    numDataSplits = TRAINING_PARAMS['NUM_SPLITS']
    print("Cross Validation used for splitting")
    for i in range(numDataSplits):
        xTrain, yTrain, xTest, yTest = splitUpDataCrossVal(x, y, numDataSplits, crossValIndex=i)
        currentAcc, yPrediction = trainModel(xTrain, yTrain, xTest, yTest)
        currentPre = calculatePrecision(xTest,yTest,yPrediction)
        pre.append(currentPre)
        acc.append(currentAcc)
        print("Accuracy with fold ", i+1, ": ", currentAcc)    
    averageAcc = np.mean(acc)
    averagePre = np.mean(pre)
else:
    print("70/30 method used for splitting")
    xTrain, yTrain, xTest, yTest =splitData7030(x, y)
    averageAcc, yPrediction = trainModel(xTrain, yTrain, xTest, yTest)
    averagePre = calculatePrecision(xTest,yTest,yPrediction)

print("Average Accuracy with Nearest Neighbors: ", averageAcc)
print("Average Precision with Nearest Neighbors: ", averagePre)
