# https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# https://www.udemy.com/deep-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range

import numpy as np
import tensorflow as tf
import q_learning


class SGDRegressor:
    def __init__(self, D):
        print("Hello TensorFlow!")
        self.lr = 0.1

        # create inputs, targets, params
        # matmul doesn't like when w is 1-D
        # so we make it 2-D and then flatten the prediction
        self.w = tf.Variable(tf.random.normal(shape=(D, 1)), name='w')
        # self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        # self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y')

    @tf.function
    def train_op(self, X, Y):
        # make prediction and cost
        Y_hat = tf.reshape(tf.matmul(X, self.w), [-1])
        delta = Y - Y_hat
        cost = tf.reduce_sum(delta * delta)
        tf.compat.v1.train.GradientDescentOptimizer(self.lr).minimize(cost, var_list=[self.w])

    @tf.function
    def predict_op(self, X):
        return tf.reshape(tf.matmul(X, self.w), [-1])

    def partial_fit(self, X, Y):
        x_tensor = tf.convert_to_tensor(X, dtype=tf.dtypes.float32)
        y_tensor = tf.convert_to_tensor(Y, dtype=tf.dtypes.float32)
        self.train_op(x_tensor, y_tensor)

    def predict(self, X):
        x_tensor = tf.convert_to_tensor(X, dtype=tf.dtypes.float32)
        return self.predict_op(x_tensor)


if __name__ == '__main__':
    q_learning.SGDRegressor = SGDRegressor
    q_learning.main()
