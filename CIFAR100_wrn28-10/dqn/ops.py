import tensorflow as tf
import math


def conv2d(x,
           output_dim,
           kernel_size,
           stride,
           activation_fn=tf.nn.relu,
           data_format='NHWC',
           padding='SAME',
           name='conv2d'):
    with tf.variable_scope(name):
        if data_format == 'NCHW':
            stride = [1, 1, stride[0], stride[1]]
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[1], output_dim]
        elif data_format == 'NHWC':
            stride = [1, stride[0], stride[1], 1]
            kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], output_dim]

        initializer_msra = tf.truncated_normal_initializer(0, math.sqrt(2./kernel_size[0]/kernel_size[1]/output_dim))
        w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer_msra)
        conv = tf.nn.conv2d(x, w, stride, padding, data_format=data_format)

        b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, b, data_format)

    if activation_fn != None:
        out = activation_fn(out)

    return out, w, b


def linear(input_, output_size, stddev=0.001, bias_start=0.0, activation_fn=None, name='linear'):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable('Matrix', [shape[1], output_size], tf.float32,
                            tf.random_normal_initializer(stddev=stddev))
        b = tf.get_variable('bias', [output_size],
                            initializer=tf.constant_initializer(bias_start))

        out = tf.nn.bias_add(tf.matmul(input_, w), b)

        if activation_fn != None:
            return activation_fn(out), w, b
        else:
            return out, w, b


def clipped_error(x):
    return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)  # Huber loss