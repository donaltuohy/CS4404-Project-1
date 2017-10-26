import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('The SUM dataset, without noise.csv', sep = ';')


X = dataset.iloc[0:1000000:, :-2].values
Y = dataset.iloc[0:1000000:,-2:-1].values

from sklearn.cross_validation import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2)


from sklearn.preprocessing import StandardScaler
sc= StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
Y_train = sc.fit_transform(Y_train)
Y_test = sc.fit_transform(Y_test)

#Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train,Y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_pred,Y_test)

print(rmse)