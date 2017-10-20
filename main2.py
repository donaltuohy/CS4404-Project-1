import tensorflow as tf
# NumPy is often used to load, manipulate and preprocess data.
import numpy as np
import matplotlib.pyplot as plt

from numpy import genfromtxt

from sklearn.datasets import load_boston


# Read in the data
# my_data = np.genfromtxt('houses_10k.csv', delimiter=',', dtype=None)
houseData = np.recfromcsv('houses_10k.csv', delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')

houseData = houseData[:100]

numInstances = len(houseData)

trainData = houseData[:round(numInstances/2)]
testData =  houseData[round(numInstances/2):]

# plt.plot(testData['sqft_living'], testData['price'], 'ro')
# plt.show()







def read_dataset(filePath,delimiter=','):
    return genfromtxt(filePath, delimiter=delimiter)

def read_boston_data():
    boston = load_boston()
    features = np.array(boston.data)
    labels = np.array(boston.target)
    return features, labels

def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma

def append_bias_reshape(features,labels):
    n_training_samples = features.shape[0]
    n_dim = features.shape[1]
    f = np.reshape(np.c_[np.ones(n_training_samples),features],[n_training_samples,n_dim + 1])
    l = np.reshape(labels,[n_training_samples,1])
    return f, l


features, outputs = read_boston_data()
normalized_features = feature_normalize(features)
features, outputs = append_bias_reshape(normalized_features, outputs)

nFeatures = features.shape[1]
nRows = features.shape[0]

rnd_indices = np.random.rand(nRows) < 0.80

train_x = features[rnd_indices]
train_y = outputs[rnd_indices]

test_x = features[~rnd_indices]
test_y = outputs[~rnd_indices]

print(train_x.shape, train_y.shape)


train_x = np.transpose(np.array([trainData['sqft_living']]))
train_y = np.transpose(np.array([trainData['price']]))


test_x = np.transpose(np.array([testData['sqft_living']]))
test_y = np.transpose(np.array([testData['price']]))

print(train_x.shape, train_y.shape)


learning_rate = 0.01
training_epochs = 1000
cost_history = np.empty(shape=[1],dtype=float)

X = tf.placeholder(tf.float32, [None, nFeatures])       # Leaving as 'None' Rows so we can use for train and predict
Y = tf.placeholder(tf.float32,[None,1])
W = tf.Variable(tf.ones([nFeatures, 1]))

init = tf.initialize_all_variables()


y_ = tf.matmul(X, W)
cost = tf.reduce_mean(tf.square(y_ - Y))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


sess = tf.Session()
sess.run(init)

for epoch in range(training_epochs):
    sess.run(training_step,feed_dict={X:train_x,Y:train_y})
    cost_history = np.append(cost_history,sess.run(cost,feed_dict={X: train_x,Y: train_y}))



# plt.plot(range(len(cost_history)),cost_history)
# plt.axis([0,training_epochs,0,np.max(cost_history)])
# plt.show()



pred_y = sess.run(y_, feed_dict={X: test_x})
mse = tf.reduce_mean(tf.square(pred_y - test_y))
print("MSE: %.4f" % sess.run(mse))

# fig, ax = plt.subplots()
# ax.scatter(test_y, pred_y)
# ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=3)
# ax.set_xlabel('Measured')
# ax.set_ylabel('Predicted')

# plt.show()
