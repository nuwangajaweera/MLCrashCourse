import tensorflow as tf

x = tf.constant(5.2)

y = tf.Variable([5])
#y = y.assign(10)

with tf.Session() as sess:
    init = tf.global_variables_initializer();
    print(y.eval())