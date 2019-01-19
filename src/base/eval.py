# coding: utf-8
from __future__ import print_function
import tensorflow as tf
import src.base.model as model
import time
import os
import cv2

tf.app.flags.DEFINE_string('loss_model', 'vgg_16', 'The name of the architecture to evaluate. '
                           'You can view all the support models in nets/nets_factory.py')
tf.app.flags.DEFINE_integer('image_size', 256, 'Image size to train.')
tf.app.flags.DEFINE_string("model_file", "./model/oil_painting.ckpt-done", "")
tf.app.flags.DEFINE_string("image_file", "./img/flower.jpg", "")

FLAGS = tf.app.flags.FLAGS


def preprocess_image(image_buffer):
    image = tf.image.decode_jpeg(image_buffer, channels=3)
    return image


def main(_):
    # Get image's height and width.
    image = cv2.imread(FLAGS.image_file)
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    tf.logging.info('Image size: %dx%d' % (width, height)) # output information

    with tf.Graph().as_default(): # create a default graph
        with tf.InteractiveSession().as_default() as sess: # create a default session
            # 取名字
            serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
            # 设置feature_configs代替placeholder
            feature_configs = {
                'input_image': tf.FixedLenFeature( # encode_image
                    shape=[], dtype=tf.string),
            }
            # 转换
            tf_example = tf.parse_example(serialized_tf_example, feature_configs)
            print("Step:1,", tf_example['input_image'])
            input_image = tf.identity(tf_example['input_image'], name='input_image')
            print("Step:2,", input_image)
            generated = tf.map_fn(preprocess_image, input_image, dtype=tf.uint8)
            generated = tf.cast(generated, tf.float32)
            print("Step:3,", generated)
            generated = model.net(generated, training=False)
            print("Step:4,", generated)
            generated = tf.cast(generated, tf.uint8)
            print("Step:5,", generated)
            # Remove batch dimension
            generated = tf.squeeze(generated, [0])
            print("Step:6,", generated)
            output_image = tf.image.encode_jpeg(generated)

            # Restore model variables.
            saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            # Use absolute path
            FLAGS.model_file = os.path.abspath(FLAGS.model_file)
            saver.restore(sess, FLAGS.model_file)
            # Make sure 'generated' directory exists.
            if os.path.exists('generated') is False:
                os.makedirs('generated')
            generated_file = 'generated/res.jpg'

            # Read image data.
            image_upload = tf.read_file(FLAGS.image_file)

            image_upload = tf.expand_dims(image_upload, 0)
            # Generate and write image data to file.
            with open(generated_file, 'wb') as img:
                start_time = time.time()
                image_ = sess.run(image_upload)

                img.write(sess.run(output_image,feed_dict={input_image:image_}))
                end_time = time.time()
                tf.logging.info('Elapsed time: %fs' % (end_time - start_time))
                tf.logging.info('Done. Please check %s.' % generated_file)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
