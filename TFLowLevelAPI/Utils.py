import os
import numpy as np
import tensorflow as tf


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