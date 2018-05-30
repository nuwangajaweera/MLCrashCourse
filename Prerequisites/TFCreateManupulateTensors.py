import tensorflow as tf


def vectorAdd():
    print("----------Vector Addition-------------")
    with tf.Graph().as_default():
        primes = tf.constant([2, 3, 5, 7, 11, 13], dtype-tf.int32)
        ones = tf.ones([6], dtype-tf.int32)
        just_beyond = tf.add(primes, ones)

        with tf.Session() as sess:
            print(just_beyond.eval())


def tensorShapes():
    print("----------Tensor Shapes---------------")
    with tf.Graph().as_default():
        scalar = tf.zeros([])
        vector = tf.zeros([3])
        matrix = tf.zeros([2,3])

        with tf.Session() as sess:
            print('Scalar has shape {0} and value {1}'.format(scalar.get_shape(), scalar.eval()))
            print('Vector has shape {0} and value {1}'.format(vector.get_shape(), vector.eval()))
            print('Matrix has shape {0} and value {1}'.format(matrix.get_shape(), matrix.eval()))


def broadcasting():
    print("----------Broadcasting----------------")
    with tf.Graph().as_default():
        primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)
        ones = tf.ones([], dtype=tf.int32)
        just_beyond = tf.add(primes, ones)

        with tf.Session() as sess:
            print(just_beyond.eval())


def matrixMultiply():
    print("----------Matrix Multiply-------------")
    with tf.Graph().as_default():
        x = tf.constant([[5, 2, 4, 3], [5, 1, 6, -2], [-1, 3, -1, -2]], dtype=tf.int32)
        y = tf.constant([[2,2], [3,5], [4,5], [1,6]], dtype=tf.int32)
        matmul_result = tf.matmul(x, y)

        with tf.Session() as sess:
            print(matmul_result.eval())


def tensorReshaping():
    print("----------Tensor Reshaping------------")
    with tf.Graph().as_default():
        mat_8x2 = tf.constant([[1,2], [3,4], [5,6], [7,8], [9,10], [11,12], [13,14], [15,16]], dtype=tf.int32)
        mat_2x8 = tf.reshape(mat_8x2, [2,8])
        mat_4x4 = tf.reshape(mat_8x2, [4,4])
        mat_1x16 = tf.reshape(mat_8x2, [1, 16])
        vector = tf.reshape(mat_8x2, [16])

        with tf.Session() as sess:
            print('mat_8x2')
            print(mat_8x2.eval())
            print('mat_2x8')
            print(mat_2x8.eval())
            print('mat_4x4')
            print(mat_4x4.eval())
            print('mat_1x16')
            print(mat_1x16.eval())
            print('vector')
            print(vector.eval())


def tensorReshapingEx():
    print("----------Tensor Reshaping (Ex)-------")
    with tf.Graph().as_default(), tf.Session() as sess:
        a = tf.constant([5, 3, 2, 7, 1, 4])
        b = tf.constant([4, 6, 3])

        a = tf.reshape(a, [2,3])
        b = tf.reshape(b, [3,1])
        matmul_result = tf.matmul(a, b)

        print('a = ')
        print(a.eval())
        print('b = ')
        print(b.eval())
        print('a x b  = ')
        print(matmul_result.eval())


def variables():
    print("----------Variables-------------------")
    with tf.Graph().as_default(), tf.Session() as sess:
        v = tf.Variable([3])
        w = tf.Variable(tf.random_normal([1], mean=1, stddev=0.35))

        initialization = tf.global_variables_initializer()
        sess.run(initialization)

        assignment = tf.assign(v, [8])

        for i in range(1,3):
            print('v = {0}, w = {1}'.format(v.eval(), w.eval()))

        sess.run(assignment)
        print('v = {0}, w = {1}'.format(v.eval(), w.eval()))


def variablesEx():
    print("----------Variables (Ex)--------------")

    with tf.Graph().as_default(), tf.Session() as sess:
        x1 = tf.Variable(tf.random_uniform([3,1], minval=1, maxval=7, dtype=tf.int32))
        x2 = tf.Variable(tf.random_uniform([3,1], minval=1, maxval=7, dtype=tf.int32))
        y = tf.add(x1, x2)

        z = tf.concat([x1, x2, y], axis=1)

        sess.run(tf.global_variables_initializer())

        print(z.eval())



def main():
    print('==========================================')
    print('Tenserflow version {0}'.format(tf.VERSION))
    #vectorAdd()
    #tensorShapes()
    #broadcasting()
    #matrixMultiply()
    #tensorReshaping()
    #tensorReshapingEx()
    #variables()
    variablesEx()
    print('=================Done=====================')


main()