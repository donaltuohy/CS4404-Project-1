from util import *
import pandas as pd
import sklearn
from sklearn import ensemble
from sklearn.ensemble import RandomForestClassifier

def trainModel(xtrain, ytrain, xtest, ytest):
    clf = RandomForestClassifier()
    clf.fit(xtrain,ytrain.reshape(len(ytrain), -1))
    accuracy = clf.score(xtest,ytest)
    return accuracy


data = readData()
x = createDesignMatrix(data)
y = createLabelVector(data)

if(TRAINING_PARAMS['SPLIT_METHOD'] == "KFOLD"):        
    acc = []
    numDataSplits = TRAINING_PARAMS['NUM_SPLITS']
    print("Cross Validation used for splitting")
    for i in range(numDataSplits):
        xTrain, yTrain, xTest, yTest = splitUpDataCrossVal(x, y, numDataSplits, crossValIndex=i)
        currentAcc = trainModel(xTrain, yTrain, xTest, yTest)
        acc.append(currentAcc)
    averageAcc = np.mean(acc)
else:
    print("70/30 method used for splitting")
    xTrain, yTrain, xTest, yTest =splitData7030(x, y)
    averageAcc = trainModel(xTrain, yTrain, xTest, yTest)



print("Accuracy with Random Forest Classification: ", averageAcc)