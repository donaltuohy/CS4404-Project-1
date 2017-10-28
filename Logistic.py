from util import *
import pandas as pd
from sklearn import linear_model

def trainModel(xtrain, ytrain, xtest, ytest):
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(xtrain, ytrain)
    yPrediction = logreg.predict(xtest)
    accuracy = logreg.score(xtest,ytest)
    return accuracy, yPrediction



data = readData()

chunkSizes = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
averageAcc = [0,0,0,0,0,0,0,0,0]
averagePre = [0,0,0,0,0,0,0,0,0]

for j in range(len(chunkSizes)):
    print("Starting learning with ", chunkSizes[j], " instances.")
    x = createDesignMatrix(data)
    y = createLabelVector(data)
    x, y = divideDataChunks(x, y, chunkSizes[j])
    if x.any() == None:
        print("Dataset doesn't contain ", chunkSizes[j], " instances.")
    else:
        print(y)
        y = multiclassToBinary(y, np.mean(y))
        if(TRAINING_PARAMS['SPLIT_METHOD'] == "KFOLD"):   
            print("Lenght of x : ", len(x))     
            acc = []
            pre = []
            numDataSplits = TRAINING_PARAMS['NUM_SPLITS']
            #print("Cross Validation used for splitting")
            for i in range(numDataSplits):
                xTrain, yTrain, xTest, yTest = splitUpDataCrossVal(x, y, numDataSplits, crossValIndex=i)
                currentAcc, yPrediction = trainModel(xTrain, yTrain, xTest, yTest)
                currentPre = calculatePrecision(xTest,yTest, yPrediction)
                acc.append(currentAcc)
                pre.append(currentPre)
            averageAcc[j] = np.mean(acc)
            averagePre[j] = np.mean(pre)
        else:
            #print("70/30 method used for splitting")
            xTrain, yTrain, xTest, yTest =splitData7030(x, y)
            averageAcc[j], yPrediction = trainModel(xTrain, yTrain, xTest, yTest)
            averagePre[j] = calculatePrecision(xTest,yTest, yPrediction)
    if x.any() == None:
        averageAcc[j] = -9999
        averagePre[j] = -9999

printMetrics(chunkSizes, averageAcc, averagePre)   