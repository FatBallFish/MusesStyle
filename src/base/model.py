import tensorflow as tf


def conv2d(x, input_filters, output_filters, kernel, strides, mode='REFLECT'):
    """
    a conv2d component of convolutional layer
    :param x: input tensor
    :param input_filters: number of input filters
    :param output_filters: number of output filters
    :param kernel: kernel size
    :param strides: strides size
    :param mode: mode of padding
    :return: tf.nn.conv2d
    """
    with tf.variable_scope('conv'):
        shape = [kernel, kernel, input_filters, output_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        x_padded = tf.pad(x, [[0, 0], [int(kernel/2), int(kernel/2)], [int(kernel/2), int(kernel/2)], [0, 0]],mode=mode)
        # REFLECT可以减轻边界效应
        return tf.nn.conv2d(x_padded, weight, strides=[1, strides, strides, 1], padding='VALID', name='conv')


def resize_conv2d(x, input_filters, output_filters, kernel, strides, training):
    """
    An alternative to transposed convolution where we first resize, then convolve.
    See http://distill.pub/2016/deconv-checkerboard/

    For some reason the shape needs to be statically known for gradient propagation
    through tf.image.resize_images, but we only know that for fixed image size, so we
    plumb through a "training" argument

    :param x: input tensor
    :param input_filters: number of input filters
    :param output_filters: number of output filters
    :param kernel: kernel size
    :param strides: strides size
    :param training: state of network, True or False
    :return: tf.nn.conv2d
    """
    with tf.variable_scope('conv_transpose'):
        height = x.get_shape()[1].value if training else tf.shape(x)[1]
        width = x.get_shape()[2].value if training else tf.shape(x)[2]

        new_height = height*strides*2
        new_width = width*strides*2
        x_resized = tf.image.resize_images(x, [new_height, new_width], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return conv2d(x_resized, input_filters, output_filters, kernel, strides)


def instance_norm(x):
    """
    a simple implementation of instance normalization
    :param x: input tensor
    :return: output tensor of instance normalization layer
    """
    epsilon = 1e-9
    mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
    return tf.div(tf.subtract(x, mean), tf.sqrt(tf.add(var, epsilon)))


def relu(x):
    """
    a relu component which can convert nan to zero
    :param x: input tensor
    :return: output tensor of relu layer
    """
    return tf.nn.relu(x)


def residual(x, filters, kernel, strides, mode=None):
    """
    a residual block component
    :param x: input tensor
    :param filters: number of filters
    :param kernel: kernel size
    :param strides: stride size
    :return:
    """
    with tf.variable_scope('residual'):
        conv1 = conv2d(x, filters, filters, kernel, strides)
        conv2 = conv2d(relu(conv1), filters, filters, kernel, strides)
        shortcut = x + conv2
    return shortcut


def net(image, training):
    """
    the generator for neural style transfer
    :param image: input image
    :param training: state of network, True or False
    :return: generated image
    """
    # Less border effects when padding a little before passing through ..
    image = tf.pad(image, [[0, 0], [10, 10], [10, 10], [0, 0]], mode='REFLECT')
    base = 16
    with tf.variable_scope('conv1'):
        conv1 = relu(instance_norm(conv2d(image, 3, 32, 9, 1)))
    with tf.variable_scope('conv2'):
        conv2 = relu(instance_norm(conv2d(conv1, 32, 64, 3, 2)))
    with tf.variable_scope('conv3'):
        conv3 = relu(instance_norm(conv2d(conv2, 64, 128, 3, 2)))
    with tf.variable_scope('res1'):
        res1 = residual(conv3, 128, 3, 1)
    with tf.variable_scope('res2'):
        res2 = residual(res1, 128, 3, 1)
    with tf.variable_scope('res3'):
        res3 = residual(res2, 128, 3, 1)
    with tf.variable_scope('res4'):
        res4 = residual(res3, 128, 3, 1)
    with tf.variable_scope('res5'):
        res5 = residual(res4, 128, 3, 1)
    with tf.variable_scope('deconv1'):
        deconv1 = relu(instance_norm(resize_conv2d(res5, 128, 64, 3, 2, training)))
    with tf.variable_scope('deconv2'):
        deconv2 = relu(instance_norm(resize_conv2d(deconv1, 64, 32, 3, 2, training)))
    with tf.variable_scope('deconv3'):
        deconv3 = tf.nn.tanh(instance_norm(conv2d(deconv2, 32, 3, 9, 1)))

    y = (deconv3+1)*127.5

    # Remove border effect reducing padding.
    height = tf.shape(y)[1]
    width = tf.shape(y)[2]
    y = tf.slice(y, [0, 10, 10, 0], tf.stack([-1, height-20, width-20, -1]))

    return y
