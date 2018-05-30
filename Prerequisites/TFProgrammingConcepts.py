import tensorflow as tf

g = tf.Graph()

with g.as_default():
    x = tf.constant(8, name='x_const')
    y = tf.constant(5, name='y_const')
    z = tf.constant(4, name='z_const')
    my_sum = tf.add(x, y, name='x_y_sum')
    my_sum_2 = tf.add(my_sum, z, name='sum_sum')

    with tf.Session() as sess:
        print(my_sum_2.eval())

g2 = tf.Graph()

with g2.as_default():
    x = tf.constant(8, name='x_const')
    y = tf.constant(5, name='y_const')
    z = tf.constant(4, name='z_const')
    my_sum = tf.add_n([x, y, z], name='x_y_z_sum')

    with tf.Session() as sess:
        print(my_sum.eval())