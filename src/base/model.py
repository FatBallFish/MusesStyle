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
        return tf.nn.conv2d(x, weight, strides=[1, strides, strides, 1], padding='SAME', name='conv')


def conv2d_transpose(x, input_filters, output_filters, kernel, strides):
    with tf.variable_scope('conv_transpose'):

        shape = [kernel, kernel, output_filters, input_filters]
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')

        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1] * strides
        width = tf.shape(x)[2] * strides
        output_shape = tf.stack([batch_size, height, width, output_filters])
        return tf.nn.conv2d_transpose(x, weight, output_shape, strides=[1, strides, strides, 1], name='conv_transpose')


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
        x_resized = tf.image.resize_images(x, [new_height, new_width], tf.image.ResizeMethod.BILINEAR)
        # x_resized = tf.image.resize_bilinear(x, [new_height, new_width], align_corners=False)
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
    relu_ = tf.nn.relu(x)
    # convert nan to zero
    # nan_to_zero = tf.where(tf.equal(relu_, relu_), relu_, tf.zeros_like(relu_))
    return relu_


def residual(x, filters, kernel, strides):
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
    image = tf.pad(image, [[0, 0], [14, 14], [14, 14], [0, 0]], mode='REFLECT')

    with tf.variable_scope('conv0'):
        conv0 = relu(instance_norm(conv2d(image, 3, 3, 3, 1))) # [?, x, y, 3]
    with tf.variable_scope('conv1'):
        image1 = tf.slice(conv0, [0, 0, 0, 0], [-1, -1, -1, 1])
        image2 = tf.slice(conv0, [0, 0, 0, 1], [-1, -1, -1, 1])
        image3 = tf.slice(conv0, [0, 0, 0, 2], [-1, -1, -1, 1])
        conv1_1 = relu(instance_norm(conv2d(image1, 1, 4, 3, 1)))
        conv1_2 = relu(instance_norm(conv2d(image2, 1, 4, 3, 1)))
        conv1_3 = relu(instance_norm(conv2d(image3, 1, 4, 3, 1)))
    with tf.variable_scope('conv2'):
        conv2_1 = relu(instance_norm(conv2d(conv1_1, 4, 8, 3, 2)))
        conv2_2 = relu(instance_norm(conv2d(conv1_2, 4, 8, 3, 2)))
        conv2_3 = relu(instance_norm(conv2d(conv1_3, 4, 8, 3, 2)))
    with tf.variable_scope('conv3'):
        conv3_1 = relu(instance_norm(conv2d(conv2_1, 8, 16, 3, 2)))
        conv3_2 = relu(instance_norm(conv2d(conv2_2, 8, 16, 3, 2)))
        conv3_3 = relu(instance_norm(conv2d(conv2_3, 8, 16, 3, 2)))
    with tf.variable_scope('res1'):
        res1_1 = residual(conv3_1, 16, 3, 1)
        res1_2 = residual(conv3_2, 16, 3, 1)
        res1_3 = residual(conv3_3, 16, 3, 1)
    with tf.variable_scope('res2'):
        res2_1 = residual(res1_1, 16, 3, 1)
        res2_2 = residual(res1_2, 16, 3, 1)
        res2_3 = residual(res1_3, 16, 3, 1)
    with tf.variable_scope('res3'):
        res3_1 = residual(res2_1, 16, 3, 1)
        res3_2 = residual(res2_2, 16, 3, 1)
        res3_3 = residual(res2_3, 16, 3, 1)
    with tf.variable_scope('res4'):
        res4_1 = residual(res3_1, 16, 3, 1)
        res4_2 = residual(res3_2, 16, 3, 1)
        res4_3 = residual(res3_3, 16, 3, 1)
    with tf.variable_scope('res5'):
        res5_1 = residual(res4_1, 16, 3, 1)
        res5_2 = residual(res4_2, 16, 3, 1)
        res5_3 = residual(res4_3, 16, 3, 1)
    with tf.variable_scope('deconv1'):
        deconv1_1 = relu(instance_norm(resize_conv2d(res5_1, 16, 8, 3, 2, training)))
        deconv1_2 = relu(instance_norm(resize_conv2d(res5_2, 16, 8, 3, 2, training)))
        deconv1_3 = relu(instance_norm(resize_conv2d(res5_3, 16, 8, 3, 2, training)))
    with tf.variable_scope('deconv2'):
        deconv2_1 = relu(instance_norm(resize_conv2d(deconv1_1, 8, 4, 3, 2, training)))
        deconv2_2 = relu(instance_norm(resize_conv2d(deconv1_2, 8, 4, 3, 2, training)))
        deconv2_3 = relu(instance_norm(resize_conv2d(deconv1_3, 8, 4, 3, 2, training)))
    with tf.variable_scope('deconv3'):
        deconv3_1 = relu(instance_norm(conv2d(deconv2_1, 4, 1, 3, 1)))
        deconv3_2 = relu(instance_norm(conv2d(deconv2_2, 4, 1, 3, 1)))
        deconv3_3 = relu(instance_norm(conv2d(deconv2_3, 4, 1, 3, 1)))
        deconv3 = tf.concat([deconv3_1, deconv3_2, deconv3_3], -1)
        print(deconv3)
    with tf.variable_scope('conv-1'):
        conv = tf.nn.tanh(instance_norm(conv2d(deconv3, 3, 3, 7, 1)))
    y = (conv+1)*127.5

    # Remove border effect reducing padding.
    height = tf.shape(y)[1]
    width = tf.shape(y)[2]
    y = tf.slice(y, [0, 14, 14, 0], tf.stack([-1, height-28, width-28, -1]))

    return y
