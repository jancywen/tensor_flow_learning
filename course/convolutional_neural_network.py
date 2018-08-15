# -*- coding: utf-8 -*-

"""

    CNN: 卷积神经网络

"""

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_sample_image


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
import matplotlib
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "cnn"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


def plot_image(image):
    plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.axis('off')


def plot_color_image(image):
    plt.imshow(image.astype(np.uint8), interpolation="nearest")
    plt.axis("off")



if __name__ == '__main__':


    reset_graph()

    china = np.array(load_sample_image('china.jpg'), dtype=np.float32)
    flower = np.array(load_sample_image('flower.jpg'), dtype=np.float32)
    dataset = np.array([china, flower], dtype=np.float32)

    batch_size, height, width, channels = dataset.shape

    filters_test = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
    filters_test[:, 3, :, 0] = 1
    filters_test[3, :, :, 1] = 1

    X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
    convolution = tf.nn.conv2d(X, filters_test, strides=[1, 2, 2, 1], padding='SAME')

    max_pool = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


    with tf.Session() as sess:
        # output = sess.run(convolution, feed_dict={X: dataset})
        output = sess.run(max_pool, feed_dict={X: dataset})


    # plt.imshow(output[0, :, :, 1], cmap='gray')
    # plt.imshow(output[0].astype(np.uint8))
    # plt.show()

    # plot_color_image(dataset[0])
    # plt.show()

    plot_color_image(output[0])
    plt.show()

"""
    # load sample images
    china = load_sample_image('china.jpg')
    image = china[150:220, 130:250]
    height, width, channels = image.shape
    image_grayscale = image.mean(axis=2).astype(np.float32)
    images = image_grayscale.reshape(1, height, width, 1)


    #
    fmap = np.zeros(shape=(7, 7, 1, 2), dtype=np.float32)
    fmap[:, 3, 0, 0] = 1
    fmap[3, :, 0, 1] = 1



    X = tf.placeholder(tf.float32, shape=(None, height, width, 1))
    feature_maps = tf.constant(fmap)
    convolution = tf.nn.conv2d(X, feature_maps, strides=[1, 1, 1, 1], padding='SAME')

    with tf.Session() as sess:
        output = convolution.eval(feed_dict={X: images})

    # plot_image(images[0, :, :, 0])
    plot_image(output[0, :, :, 1])
    plt.show()

"""


