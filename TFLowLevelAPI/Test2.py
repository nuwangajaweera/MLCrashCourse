import os
import numpy as np
import tensorflow as tf

import Utils as myutils


def main():
    myutils.start_banner()

    vec1 = tf.random_uniform(shape=[3])

    x = tf.placeholder(tf.float32, shape=[3], name='my_x')
    y = tf.placeholder(tf.float32, name='my_y')
    z = x + y
    z1 = vec1 + z

    with tf.Session() as sess:
        output = sess.run({'z1': z1, 'vec1': vec1, 'z': z}, feed_dict={x: [1.0, 2.0, 3.0], y: 2.0})
        for key in output:
            print('{0} : {1}'.format(key, output[key]))

        # print('vec1 = {0}, z1 = {1}, z = {2}'.format(vec1.eval(), z1.eval(), z.eval()))
        # print('z = {0}'.format(z.eval()))

    myutils.dump_graph(tf.get_default_graph())


    myutils.end_banner()


main()