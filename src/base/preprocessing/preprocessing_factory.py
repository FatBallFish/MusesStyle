from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from src.base.preprocessing import vgg_preprocessing

slim = tf.contrib.slim


def get_preprocessing(is_training=False):
    """Returns preprocessing_fn(image, height, width, **kwargs).

    Args:
      is_training: `True` if the model is being used for training and `False`
        otherwise.

    Returns:
      preprocessing_fn: A function that preprocessing a single image (pre-batch).
        It has the following signature:
          image = preprocessing_fn(image, output_height, output_width, ...).
    """

    def preprocessing_fn(image, output_height, output_width, **kwargs):
        return vgg_preprocessing.preprocess_image(
            image, output_height, output_width, is_training=is_training, **kwargs)

    def unprocessing_fn(image, **kwargs):
        return vgg_preprocessing.unprocess_image(
            image, **kwargs)

    return preprocessing_fn, unprocessing_fn
