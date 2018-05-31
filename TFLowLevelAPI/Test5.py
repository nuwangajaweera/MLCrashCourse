import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets as ds
import tensorflow as tf

import Utils as myutils


def start_banner():
    print('===============================')
    print('Script File        : {0}'.format(__file__))
    print('Numpy version      : {0}'.format(np.__version__))
    print('Tenserflow version : {0}'.format(tf.VERSION))
    print('===============================')


def end_banner():
    print('============ Done =============')


def dump_graph(my_graph):
    writer = tf.summary.FileWriter('./logs')
    writer.add_graph(my_graph)
    cwd = os.getcwd()
    print('Dump default graph to dir : {0}'.format(cwd))
    writer.close()


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

    [features, target, _] = load_dataset()

    # dt = pd.DataFrame(data=features, columns=feature_names)

    slices = tf.data.Dataset.from_tensor_slices(features)
    next_item = tf.data.Dataset.make_one_shot_iterator(slices).get_next()

    # with tf.Session() as sess:
    #     for i in range(10):
    #         x = sess.run(next_item)
    #         print('shape : {0}, data = {1}'.format(x.shape, x))

    # x_val = [
    #     [1., 1., 1.], [2., 2., 2.], [3., 3., 3.], [4., 4., 4.]
    # ]
    x_val = [[1., 2.], [2., 1.], [3., 0.], [4., -1.]]
    y_val = [[1.], [-0.], [-1.], [-2.]]

    xx_val = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]

    # x_val = features[:, 1].reshape([5, 1])
    # y_val = target.reshape([5, 1])

    x = tf.placeholder(tf.float32, shape=[None, 2], name='x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y')
    lin_model = tf.layers.Dense(units=1, use_bias=False)
    y_pred = lin_model(x)

    loss = tf.losses.mean_squared_error(labels=y, predictions=y_pred)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    weights = []
    loss_history = []
    training_iterations = range(200)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in training_iterations:
            _, loss_val = sess.run((train, loss), {x: x_val, y:y_val})
            loss_history.append(loss_val)

        y_pred_val = sess.run(y_pred, {x: x_val})
        weights = sess.run(lin_model.trainable_variables)

        for i in range(4):
            print(y_val[i], y_pred_val[i])

    print(weights)
    #plot_learning_rate(loss_history)

    # my_data = [
    #     [0, 1],
    #     [2, 3],
    #     [4, 5],
    #     [6, 7],
    # ]
    #
    # slices = tf.data.Dataset.from_tensor_slices(my_data)
    # next_item = slices.make_one_shot_iterator().get_next()
    #
    # with tf.Session() as sess:
    #     while True:
    #         try:
    #             print('Slice {0}'.format(sess.run(next_item)))
    #         except tf.errors.OutOfRangeError:
    #             break
    #
    # with tf.Session() as sess:
    #     r = tf.random_normal([10, 3])
    #     dataset = tf.data.Dataset.from_tensor_slices(r)
    #     iterator = dataset.make_initializable_iterator()
    #     next_row = iterator.get_next()
    #
    #     sess.run(iterator.initializer)
    #     while True:
    #         try:
    #             print(sess.run(next_row))
    #         except tf.errors.OutOfRangeError:
    #             break

    myutils.dump_graph(tf.get_default_graph())

    myutils.end_banner()


main()