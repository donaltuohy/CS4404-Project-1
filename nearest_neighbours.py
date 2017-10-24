from util import *
import pandas as pd
from sklearn import neighbors
from sklearn.neighbors import NearestNeighbors

n_neighbors = 1

data = readData()

x = createDesignMatrix(data)
y = createLabelVector(data)

xtrain, ytrain, xtest, ytest = splitData(x,y)

clf = neighbors.KNeighborsClassifier()
clf.fit(xtrain, ytrain.reshape(len(ytrain),1))

accuracy = clf.score(xtest,ytest)
print(accuracy)

prediction = clf.predict(xtest)
