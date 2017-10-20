import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np

import matplotlib.pyplot as plot

# Read in the data
# my_data = np.genfromtxt('houses_10k.csv', delimiter=',', dtype=None)
houseData = np.recfromcsv('houses_10k.csv', delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')

houseData = houseData[:100]

numInstances = len(houseData)

trainData = houseData[:round(numInstances/2)]
testData =  houseData[round(numInstances/2):]

plot.plot(testData['sqft_living'], testData['price'], 'ro')
plot.show()

learning_rate = 0.01
training_epochs = 1000
n_dim = 1






print(X, y, w)

init = tf.initialize_all_variables()

y_ = tf.matmul(X, w)
cost = tf.reduce_mean(tf.square(y_ - y))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    sess.run(training_step,feed_dict={X:trainData['sqft_living'],y:trainData['price']})
    cost_history = np.append(cost_history,sess.run(cost,feed_dict={X: trainData['sqft_living'],y: trainData['price']}))







