import os
import numpy as np
import tensorflow as tf

import Utils as myutils


def main():
    myutils.start_banner()

    a = tf.constant(3.0, dtype=tf.float32)
    b = tf.constant(4.0, name='const_b')
    c = tf.constant(5.0, name='const_c')
    d = tf.constant(6.0, name='const_d')
    a_plus_b = tf.add(a, b, name='a_plus_b')
    tot_sum  = tf.add(a_plus_b, c, name='tot_sum')
    tot_sum2 = tf.add(tot_sum, d, name='tot_sum2')
    print('a = {0}'.format(a))
    print('b = {0}'.format(b))
    print('c = {0}'.format(c))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run({'abc': (a, b, c), 'a_plus_b': a_plus_b, 'tot_sum, tot_sum2': (tot_sum, tot_sum2)}))
        # print('a = {0}'.format(a.eval()))
        # print('b = {0}'.format(b.eval()))
        # print('c = {0}'.format(c.eval()))
        # print('a_plus_b = {0}'.format(a_plus_b.eval()))
        # print('tot_sum = {0}'.format(tot_sum.eval()))

    myutils.dump_graph(tf.get_default_graph())

    myutils.end_banner()


main()
