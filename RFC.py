from util import *
import pandas as pd
import sklearn
from sklearn import neighbors, ensemble
from sklearn.ensemble import RandomForestClassifier

n_neighbors = 1

data = readData()
print("Data read in")

x = createDesignMatrix(data)
y = createLabelVector(data)
print("Vectors created")

xtrain, ytrain, xtest, ytest = splitData(x,y)
print("Data split")

clf = RandomForestClassifier()
print("about to fit")

clf.fit(xtrain,ytrain.reshape(len(ytrain), -1))
print("Data Fitted")

accuracy = clf.score(xtest,ytest)
print("Accuracy with Random Forest Classification: ", accuracy)

prediction = clf.predict(xtest)
print(prediction)