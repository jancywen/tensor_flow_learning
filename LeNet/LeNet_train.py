# -*- coding: utf-8 -*-


import os

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from LeNet import LeNet_inference

from tensorflow.contrib import layers

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAIN_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'model3.ckpt'


def train(mnist):
    x = tf.placeholder(tf.float32, [BATCH_SIZE, LeNet_inference.IMAGE_SIZE,
                                    LeNet_inference.IMAGE_SIZE, LeNet_inference.NUM_CHANNELS], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, LeNet_inference.OUTPUT_NODE], name='y-input')

    regularizer = layers.l2_regularizer(REGULARAZTION_RATE)

    y = LeNet_inference.inference(x, train, regularizer)

    global_step = tf.Variable(0, trainable=False)

    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step=global_step,
                                               decay_steps=mnist.train.num_examples / BATCH_SIZE,
                                               decay_rate=LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(TRAIN_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            xs = np.reshape(xs, [BATCH_SIZE, LeNet_inference.IMAGE_SIZE,
                                 LeNet_inference.IMAGE_SIZE,
                                 LeNet_inference.NUM_CHANNELS])
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training steps, loss on training"
                      "batch is %g" % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv=None):
    mnist = input_data.read_data_sets("E:\科研\TensorFlow教程\MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
