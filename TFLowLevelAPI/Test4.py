import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets as ds
import tensorflow as tf

import Utils as myutils


def load_dataset(verbose=False):
    data = ds.load_boston()

    if(verbose):
        for key in data:
            print('{0} : {1}'.format(key, type(data[key])))

    features = data['data']
    target = data['target']
    feature_names = data['feature_names']

    print('Loaded : features.shape = {0}, lables.shape = {1}'.format(features.shape, target.shape))

    if(verbose):
        print('features = {0}'.format(feature_names))
        print('description = {0}'.format(data['DESCR']))

    # reduce features to simplify debug
    used_feature_idx = [5, 7, 8]

    # reduce sample count to simplify debug
    used_sample_count = 5

    feature_names = feature_names[used_feature_idx]
    features = features[:used_sample_count, used_feature_idx]
    target = target[:used_sample_count]

    print('Used features : {0}'.format(feature_names))
    print('Used : features.shape = {0}, target.shape = {1}'.format(features.shape, target.shape))

    return features, target, feature_names


def plot_learning_rate(loss_history):
    # plt.plot(training_iterations, loss_history)
    plt.plot(loss_history)
    plt.xlabel('iterations/epocs')
    plt.ylabel('loss')
    plt.grid(True)
    plt.show()


def plot_1d_model(x, y, y_pred):
    plt.plot(x, y, 'x')
    plt.plot(x, y_pred)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()


def main():
    myutils.start_banner()

    x_val = [
        [1.], [2.], [3.], [4.]
    ]
    y_val = [
        [0.], [-1.], [-2.], [-3.]
    ]

    x = tf.placeholder(tf.float32, shape=[None, 1])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    lin_model = tf.layers.Dense(units=1)
    y_pred = lin_model(x)

    loss = tf.losses.mean_squared_error(labels=y, predictions=y_pred)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    loss_history = []
    training_iterations = range(200)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in training_iterations:
            _, loss_val = sess.run((train, loss), {x: x_val, y:y_val})
            loss_history.append(loss_val)

        y_pred_val = sess.run(y_pred, {x: [[1.0], [2.0], [3.0], [4.0]]})

        for i in range(3):
            print(y_val[i], y_pred_val[i])

    plot_learning_rate(loss_history)

    myutils.dump_graph(tf.get_default_graph())

    myutils.end_banner()


main()