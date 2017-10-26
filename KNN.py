from util import *
import pandas as pd
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors

def trainModel(xtrain, ytrain, xtest, ytest):
    clf = neighbors.KNeighborsClassifier()
    clf.fit(xtrain, ytrain)
    accuracy = clf.score(xtest,ytest)
    return accuracy

data = readData()

chunkSizes = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
averageAcc = [0,0,0,0,0,0,0,0,0]

for j in range(len(chunkSizes)):
    print("Starting learning with ", chunkSizes[j], " instances.")
    x = createDesignMatrix(data)
    y = createLabelVector(data)
    x, y = divideDataChunks(x, y, chunkSizes[j])
    if x.any() == None:
        print("Dataset doesn't contain ", chunkSizes[j], " instances.")
    else:
        if(TRAINING_PARAMS['SPLIT_METHOD'] == "KFOLD"):        
            acc = []
            numDataSplits = TRAINING_PARAMS['NUM_SPLITS']
            #print("Cross Validation used for splitting")
            for i in range(numDataSplits):
                xTrain, yTrain, xTest, yTest = splitUpDataCrossVal(x, y, numDataSplits, crossValIndex=i)
                currentAcc = trainModel(xTrain, yTrain, xTest, yTest)
                acc.append(currentAcc)
            averageAcc[j] = np.mean(acc)
        else:
            #print("70/30 method used for splitting")
            xTrain, yTrain, xTest, yTest =splitData7030(x, y)
            averageAcc[j] = trainModel(xTrain, yTrain, xTest, yTest)
    if x.any() == None:
        averageAcc[j] = -9999

for j in range(len(averageAcc)):
    print("Accuracy with ", chunkSizes[j], "is : ", averageAcc[j])
       