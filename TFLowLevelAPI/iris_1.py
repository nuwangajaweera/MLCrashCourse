import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets as ds
import tensorflow as tf
import seaborn as sns
import Utils as myutils

TEST_SET_PERCENTAGE = 0.1


def start_banner():
    print('\n----------------\n')


def end_banner():
    print('\n----------------\n')


def load_iris_data():

    def gen_classification(row):
        if row['species'] == 'versicolor':
            return 1
        elif row['species'] == 'virginica':
            return 2
        else:
            return 0 # assume 'setosa'

    # load data
    training_set = sns.load_dataset('iris')
    total_samples = len(training_set)

    # since one column is the classification/target
    feature_cols = training_set.num_columns - 1

    # generate integer classification
    training_set['target'] = training_set.apply(gen_classification, axis=1)

    # create empty traiing set
    test_set = pd.DataFrame(columns=list(training_set))

    # partition data set in to test set and training set
    for i in range(3):
        matching_indices = training_set.index[training_set['target'] == i].tolist()
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


def main():
    start_banner()

    training_set, test_set, feature_cols = load_iris_data()

    x_dims = feature_cols
    y_dims = 1

    X = tf.placeholder(tf.float32, [None, x_dims], name='X')
    y = tf.placeholder(tf.float32, [None, 1], name='y')
    liner_model = tf.layers.Dense(units=1, activation=)
    y_bar = liner_model(X)
    y_pred = tf.nn.sigmoid(y_bar, name='sigmoid')

    loss = tf.losses.log_loss

    end_banner()
main()

