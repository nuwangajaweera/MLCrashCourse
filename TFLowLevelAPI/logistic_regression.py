import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets as ds
import tensorflow as tf
import seaborn as sns
import Utils as myutils

TEST_SET_PERCENTAGE = 1/3
TRAINING_ITERATIONS = 100


def start_banner():
    print('\n----------------\n')


def end_banner():
    print('\n----------------\n')


def load_data():
    # load data
    #x, y = ds.make_classification(200, n_features=2, n_classes=2, n_clusters_per_class=1, n_redundant=0)
    x, y = ds.make_classification(200, n_features=2, n_classes=2, n_redundant=0)
    data = np.column_stack((x,y))
    training_set = pd.DataFrame(data, columns=['x1', 'x2', 'label'])

    total_samples = len(training_set)
    feature_cols = len(training_set.columns) - 1

    # create empty training set
    test_set = pd.DataFrame(columns=list(training_set))

    # partition data set in to test set and training set
    for i in range(2):
        matching_indices = training_set.index[training_set['label'] == i].tolist()
        np.random.shuffle(matching_indices)
        size = len(matching_indices)
        testset_size = round(size * TEST_SET_PERCENTAGE)
        testset_indices = matching_indices[:testset_size]
        test_subset = training_set.loc[testset_indices]
        test_set = test_set.append(test_subset)
        training_set = training_set.drop(index=testset_indices)

    # suffle training set
    training_set = training_set.reindex(np.random.permutation(training_set.index))

    print('Loaded {0} samples'.format(total_samples))
    print('\tTraining Set size = {0} samples, Test Set Size = {1} samples'.format(len(training_set), len(test_set)))

    return training_set, test_set, feature_cols


def plot_learning_rate(ax, loss_trend):
    ax.plot(loss_trend)
    ax.set_yscale('log')
    ax.set_xlabel('iterations/epocs')
    ax.set_ylabel('loss')
    ax.grid(True)


def plot_model(ax, a, c, x, y):
    x1 = x[:, 0]
    x2 = x[:, 1]
    y = y[:, 0]

    x1_lims = np.array([np.amin(x1), np.amax(x1)])
    x2_lims = (x1_lims * a[0] - c) / a[1]

    ax.scatter(x1, x2, c=y)
    ax.plot(x1_lims, x2_lims)


def main():
    start_banner()

    training_set, test_set, feature_cols = load_data()

    x_dims = feature_cols
    y_dims = 1

    x = tf.placeholder(tf.float32, [None, x_dims], name='x')
    y = tf.placeholder(tf.float32, [None, 1], name='y')
    model = tf.layers.Dense(units=1, activation=tf.nn.sigmoid)
    y_pred = model(x)

    loss = tf.losses.log_loss(y, y_pred)
    optimizrer = tf.train.GradientDescentOptimizer(0.3)
    train = optimizrer.minimize(loss)

    y_vals = np.reshape(training_set['label'].values, [len(training_set), 1])
    x_vals = training_set.drop('label', axis=1).values
    # print(np.shape(x_vals), np.shape(y_vals))

    y_test_vals = np.reshape(test_set['label'].values, [len(test_set), 1])
    x_test_vals = test_set.drop('label', axis=1).values

    loss_trend = []
    weights = []

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(TRAINING_ITERATIONS):
            _, current_loss_val = sess.run((train, loss), {x: x_vals, y: y_vals})
            loss_trend.append(current_loss_val)
            if i % 20 == 0:
                print('@{0} iterations loss = {1}'.format(i, current_loss_val))

        weights = sess.run(model.trainable_variables)
        y_pred_vals = sess.run(y_pred, {x:x_test_vals})

    y_pred_vals = np.where(y_pred_vals > 0.5, 1, 0)

    correct_predictions = np.add.reduce((y_pred_vals == y_test_vals).astype(dtype=np.int))[0]
    accuracy = correct_predictions / len(test_set)

    print('Accuracy based on test set = {0:6.2f} %'.format(accuracy*100))

    ax = plt.subplot('131')
    plot_learning_rate(ax, loss_trend)

    a = weights[0]
    c = weights[1]
    ax = plt.subplot('132')
    plot_model(ax, a, c, x_vals, y_vals)

    ax = plt.subplot('133')
    plot_model(ax, a, c, x_test_vals, y_test_vals)

    plt.show()

    end_banner()


main()

