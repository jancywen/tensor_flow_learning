# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np


os.environ['TF_CNN_MIN_LOG_LEVEL'] = '2'

def reset_graph(seed=42):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    tf.reset_default_graph()



height = 28
width = 28
channels = 1
n_inputs = 28 * 28


conv1_fmaps = 32
conv1_filter = 3
conv1_stride = 1
conv1_pad = 'SAME'

conv2_fmaps = 64
conv2_filter = 3
conv2_stride = 1
conv2_pad = 'SAME'
conv2_dropout_rate = 0.25

pool3_fmap = conv2_fmaps

n_fc1 = 64
fc1_dropout_rate = 0.5

n_outputs = 10



with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name='X')
    X_reshape = tf.reshape(X,  shape=(None, height, width, channels))
    y = tf.placeholder(tf.int32, shape=[None], name='y')
    training = tf.placeholder_with_default(False,shape=[], name='training')


conv1 = tf.layers.conv2d(X_reshape, conv1_fmaps, conv1_filter, conv1_stride, padding=conv1_pad,
                         activation=tf.nn.relu, name='conv1')
conv2 = tf.layers.conv2d(conv1, conv2_fmaps, kernel_size=conv2_filter, strides=conv2_stride,
                         padding=conv2_pad, activation=tf.nn.relu, name='conv2')


with tf.name_scope('pool3'):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmap * 14 * 14])
    pool3_flat_drop = tf.layers.dropout(pool3_flat,rate=conv2_dropout_rate, training=training)

with tf.name_scope('fc1'):
    fc1 = tf.layers.dense(pool3_flat_drop, n_fc1, activation=tf.nn.relu, name='fc1')
    fc1_drop = tf.layers.dropout(fc1, rate=fc1_dropout_rate, training=training)

with tf.name_scope('output'):
    logists = tf.layers.dense(fc1, n_outputs, name='output')
    Y_prod = tf.nn.softmax(logists, name='Y_prod')

with tf.name_scope('train'):
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logists, labels=y)
    loss = tf.reduce_mean(entropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logists, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope('init_and_save'):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/')


def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvars.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}

def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + '/Assign') for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)



n_epochs = 1000
batch_size = 50
best_loss_val = np.infty
check_interval = 500
checks_since_last_progress = 0
max_checks_without_progress = 20
best_model_params = None


"""
每100次训练迭代，它在验证集上评估模型，
如果模型的性能比目前发现的最佳模型更好，那么它会将模型保存到RAM中，
如果连续100次评估没有进展，那么培训就会中断，
经过培训后，代码将恢复找到的最佳模型。
"""

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for interaction in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

            if interaction % check_interval == 0:
                loss_avl = loss.eval(feed_dict={X: mnist.validation.images, y: mnist.validation.labels})
                if loss_avl < best_loss_val:
                    best_loss_val = loss_avl
                    checks_since_last_progress = 0
                    best_model_params = get_model_params()
                else:
                    checks_since_last_progress += 1
        acc_train = accuracy .eval(feed_dict={X: X_batch, y: y_batch})
        acc_val = accuracy.eval(feed_dict={X: mnist.validation.images, y: mnist.validation.labels})

        print('Epoch {}, train accuracy: {:.4f}%, valid.accuracy: {:.4f}%, valid.best loss: {:.6f}'.format(
            epoch, acc_train * 100, acc_val * 100, best_loss_val))

        if best_model_params:
            restore_model_params(best_model_params)
        acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
        print('Final accuracy on test set: ', acc_test)
