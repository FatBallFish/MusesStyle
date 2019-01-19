from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import tensorflow as tf
from src.base.nets import vgg

slim = tf.contrib.slim


def get_network_fn(num_classes, weight_decay=0.0, is_training=False):
    """Returns a network_fn such as `logits, end_points = network_fn(images)`.

    Args:
      num_classes: The number of classes to use for classification.
      weight_decay: The l2 coefficient for the model weights.
      is_training: `True` if the model is being used for training and `False`
        otherwise.

    Returns:
      network_fn: A function that applies the model to a batch of images. It has
        the following signature:
          logits, end_points = network_fn(images)
    Raises:
      ValueError: If network `name` is not recognized.
    """
    arg_scope = vgg.vgg_arg_scope(weight_decay=weight_decay)
    func = vgg.vgg_16

    @functools.wraps(func)
    def network_fn(images, **kwargs):
        with slim.arg_scope(arg_scope):
            return func(images, num_classes, is_training=is_training, **kwargs)

    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn
