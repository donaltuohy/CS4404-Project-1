import tensorflow as tf
import numpy as np


x = tf.placeholder(tf.float32, shape=(2, 2))
y = tf.matmul(x, x)

with tf.Session() as sess:.
  print(sess.run(y, feed_dict={x: [[1,2], [1,2]}))  # Will succeed.