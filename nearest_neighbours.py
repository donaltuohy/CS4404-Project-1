from util import *
import pandas as pd
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors

n_neighbors = 1

data = readData()
print("Data read in")
x = createDesignMatrix(data)
y = createLabelVector(data)
print("Vectors created")
xtrain, ytrain, xtest, ytest = splitData(x,y)
print("Data Split")
clf = neighbors.KNeighborsClassifier()
print("About to fit")
clf.fit(xtrain, ytrain)
print("fitted data")
accuracy = clf.score(xtest,ytest)
print(accuracy)

prediction = clf.predict(xtest)
