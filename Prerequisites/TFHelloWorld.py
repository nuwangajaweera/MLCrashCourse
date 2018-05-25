import tensorflow as tf

msg = tf.constant('Hello World')

#sess = tf.Session()
#print(sess.run(msg))

with tf.Session() as sess:
    print(sess.run(msg))