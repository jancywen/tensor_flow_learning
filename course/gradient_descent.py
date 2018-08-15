# -*- coding: utf-8 -*-
__author__ = 'wangwenjie'
__data__ = '2018/6/14 下午4:24'
__product__ = 'PyCharm'
__filename__ = 'gradient_descent'


""" 梯度下降 """

"""Gradient Descent requires scaling the feature vectors first."""

"""
    1, linear regression
    1, manually computing
    2, Using autodiff   tf.gradients函数
    3, GradientDescentOptimizer
"""


import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
housing = fetch_california_housing()
scaled_housing_data = StandardScaler().fit_transform(housing.data)
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

'''
from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = 'tf_logs'
logdir = "{}/run-{}".format(root_logdir, now)


def reset_graph(seed=42):
    """
    重置图
    :param seed:
    :return:
    """
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


'''
def linear_regression_normal_equation():
    """
    Using the Normal Equation
    :return: 
    """
    
    X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")


    XT = tf.transpose(X)
    theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

    with tf.Session() as sess:
        theta_value = theta.eval()
        print(theta_value)
'''


'''
def manually_computing():

    """
    using manually computing 
    :return: 
    """
    
    n_epochs = 1000
    
    X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
    y_pred = tf.matmul(X, theta, name='predictions')
    error = y_pred - y

    mse = tf.reduce_mean(tf.square(error), name='mse')

    gradients = 2/m * tf.matmul(tf.transpose(X), error)
    training_op = tf.assign(theta, theta - learning_rate * gradients)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print('Epoch', epoch, 'MSE=', mse.eval())
            sess.run(training_op)
        best_theta = theta.eval()
        print(best_theta)

'''

'''
def using_autodiff():
    """
    using tf.gradients 
    :return: 
    """
    
    n_epochs = 1000
    
    X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
    y_pred = tf.matmul(X, theta, name='predictions')
    error = y_pred - y

    mse = tf.reduce_mean(tf.square(error), name='mse')

    gradients = tf.gradients(mse, [theta])[0]
    training_op = tf.assign(theta, theta - learning_rate * gradients)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print('Epoch', epoch, 'MSE=', mse.eval())
            sess.run(training_op)
        best_theta = theta.eval()
        print(best_theta)
'''

'''
def using_gradient_descent_optimizer():
    """
    using GradientDescentOptimizer
    :return:
    """
    
    n_epochs = 1000
    learning_rate = 0.01

    X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")

    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name='theta')
    y_pred = tf.matmul(X, theta, name='predictions')
    error = y_pred - y

    mse = tf.reduce_mean(tf.square(error), name='mse')

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print('Epoch', epoch, 'MSE=', mse.eval())
            sess.run(training_op)
            # print(theta.eval())

        best_theta = theta.eval()
        print(best_theta)
'''
'''
def mini_batch_gradient_descent():
    
    """
    feed data to the training algorithm
    :return: 
    """
    n_epochs = 1000
    learning_rate = 0.01

    X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name='theta')
    y_pred = tf.matmul(X, theta, name='predictions')
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name='mse')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()

    n_epochs = 10
    batch_size = 100
    n_batches = int(np.ceil(m / batch_size))


    def fetch_batch(epoch, batch_index, batch_size):
        np.random.seed(epoch * n_batches + batch_size)
        indices = np.random.randint(m, size=batch_size)
        X_batch = housing_data_plus_bias[indices]
        y_batch = housing.target.reshape(-1, 1)[indices]
        return X_batch, y_batch

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        best_theta = theta.eval()
        print(best_theta)

'''

'''
def save_a_model():
    """
    保存模型计算节点
    :return:
    """

    n_epochs = 1000
    learning_rate = 0.01


    X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
    y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name='theta')
    y_pred = tf.matmul(X, theta, name='predictions')
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name='mse')
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:

        # saver.restore(sess, './tmp/my_model.ckpt')
        # print(theta.eval())

        sess.run(init)

        for epoch in range(n_epochs):
            if epoch % 100 == 0:
                print('Epoch', epoch, 'MSE=', mse.eval())
                # 需要新建一个tmp文件夹
                save_path = saver.save(sess, './tmp/my_model.ckpt')

            sess.run(training_op)

        best_theta = theta.eval()
        save_path = saver.save(sess, './tmp/my_model_final.ckpt')

        print()
'''

'''

def restore_a_model():
    saver = tf.train.import_meta_graph('./tmp/my_model.ckpt.meta')
    theta = tf.get_default_graph().get_tensor_by_name('theta:0')
    training_op = tf.get_default_graph().get_operation_by_name('training_op')

    with tf.Session() as sess:
        saver.restore(sess, './tmp/my_model.ckpt')
        for epoch in range(400):
            sess.run(training_op)
        print(theta.eval())
'''

'''
def visualizing_graph():
    n_epochs = 1000
    learning_rate = 0.01

    X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
    theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name='theta')
    y_pred = tf.matmul(X, theta, name='predictions')

    with tf.name_scope('loss') as scope:
        error = y_pred - y
        mse = tf.reduce_mean(tf.square(error), name='mse')

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(mse)

    init = tf.global_variables_initializer()

    mse_summary = tf.summary.scalar('MSE', mse)
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    n_epochs = 10
    batch_size = 100
    n_batches = int(np.ceil(m / batch_size))

    def fetch_batch(epoch, batch_index, batch_size):
        np.random.seed(epoch * n_batches + batch_size)
        indices = np.random.randint(m, size=batch_size)
        X_batch = housing_data_plus_bias[indices]
        y_batch = housing.target.reshape(-1, 1)[indices]
        return X_batch, y_batch

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)

                if batch_index % 10 == 0:
                    summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                    step = epoch * n_batches + batch_index
                    file_writer.add_summary(summary_str, step)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        best_theta = theta.eval()
        file_writer.flush()
        file_writer.close()
        print(best_theta)
'''

def visualizing_graph_modularity():
    # a1 = tf.Variable(0, name='a')
    # a2 = tf.Variable(0, name='b')
    #
    # with tf.name_scope('param'):
    #     a3 = tf.Variable(0, name='a')
    # with tf.name_scope('param'):
    #     a4 = tf.Variable(0, name='b')
    #
    # for node in (a1, a2, a3, a4):
    #     print(node.op.name)

    n_features = 3
    X = tf.placeholder(tf.float32, shape=(None, n_features), name='X')

    # w1 = tf.Variable(tf.random_normal((n_features, 1)), name='weights1')
    # w2 = tf.Variable(tf.random_normal((n_features, 1)), name='weights2')
    # b1 = tf.Variable(0.0, name='bias1')
    # b2 = tf.Variable(0.0, name='bias2')
    #
    # z1 = tf.add(tf.matmul(X, w1), b1, name='z1')
    # z2 = tf.add(tf.matmul(X, w2), b2, name='z2')
    #
    # relu1 = tf.maximum(z1, 0, name='relu1')
    # relu2 = tf.maximum(z2, 0, name='relu2')
    #
    # output = tf.add(relu1, relu2, name='output')


    def relu(X):
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name='weights')
        b = tf.Variable(0.0, name='bias')
        z = tf.add(tf.matmul(X, w), b, name='z')
        return tf.maximum(z, 0., name='relu')

    relus = [relu(X) for _ in range(5)]
    output = tf.add_n(relus, name='output')

    file_wirte = tf.summary.FileWriter('tf_logs/relu1', tf.get_default_graph())
    file_wirte.close()
    pass


if __name__ == '__main__':
    reset_graph()

    # linear_regression_normal_equation()

    # manually_computing()

    # using_autodiff()

    # using_gradient_descent_optimizer()

    # mini_batch_gradient_descent()

    # save_a_model()

    # visualizing_graph()

    visualizing_graph_modularity()
