# coding: utf-8
from __future__ import print_function
import tensorflow as tf
from src.base.nets import nets_factory
from src.base.preprocessing import preprocessing_factory
import src.base.utils as utils
import os
import time
slim = tf.contrib.slim


def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    grams = tf.matmul(filters, filters, transpose_a=True)/tf.to_float(width*height*num_filters)

    return grams


def get_style_features(filterInfo, Flags, debug):

    with tf.Graph().as_default():
        network_fn = nets_factory.get_network_fn(num_classes=1, is_training=False)
        image_preprocess_fn, image_unprocess_fn = preprocessing_factory.get_preprocessing(is_training=False)

        # Get the style image data
        size = filterInfo.brush_size
        img_bytes = tf.decode_base64(filterInfo.style_template)
        image = tf.image.decode_image(img_bytes)

        # Add the batch dimension
        images = tf.expand_dims(image_preprocess_fn(image, size, size), 0)

        _, endpoints_dict = network_fn(images, spatial_squeeze=False)
        features = []

        for layer in Flags.style_layers:
            feature = endpoints_dict[layer]
            feature = tf.squeeze(gram(feature), [0])  # remove the batch dimension
            features.append(feature)

        with tf.Session() as sess:
            # Restore variables for loss network.
            init_func = utils._get_init_fn(Flags)
            init_func(sess)

            if debug:
                # Make sure the 'generated' directory is exists.
                if os.path.exists(Flags.target) is False:
                    os.makedirs(Flags.target)
                # Indicate cropped style image path
                save_file = Flags.target + str(int(time.mktime(filterInfo.upload_time.timetuple())))+'.jpg'
                # Write preprocessed style image to indicated path
                with open(save_file, 'wb') as f:
                    target_image = image_unprocess_fn(images[0, :])
                    value = tf.image.encode_jpeg(tf.cast(target_image, tf.uint8))
                    f.write(sess.run(value))
                    print('Target style pattern is saved to: %s.'%save_file)

            # Return the features those layers are use for measuring style loss.
            return sess.run(features)


def style_loss(endpoints_dict, style_features_t, style_layers):
    style_loss = 0
    style_loss_summary = {}
    for style_gram, layer in zip(style_features_t, style_layers):
        generated_images, _ = tf.split(endpoints_dict[layer], 2, 0)
        size = tf.size(generated_images)
        layer_style_loss = tf.nn.l2_loss(gram(generated_images)-style_gram)*2/tf.to_float(size)
        style_loss_summary[layer] = layer_style_loss
        style_loss += layer_style_loss
    return style_loss, style_loss_summary


def content_loss(endpoints_dict, content_layers):
    content_loss = 0
    for layer in content_layers:
        generated_images, content_images = tf.split(endpoints_dict[layer], 2, 0)
        size = tf.size(generated_images)
        content_loss += tf.nn.l2_loss(generated_images-content_images)*2/tf.to_float(size)
        # remain the same as in the paper
    return content_loss


def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, height-1, -1, -1]))-tf.slice(layer, [0, 1, 0, 0], [-1, -1, -1, -1])
    x = tf.slice(layer, [0, 0, 0, 0], tf.stack([-1, -1, width-1, -1]))-tf.slice(layer, [0, 0, 1, 0], [-1, -1, -1, -1])
    loss = tf.nn.l2_loss(x)/tf.to_float(tf.size(x))+tf.nn.l2_loss(y)/tf.to_float(tf.size(y))
    return loss
