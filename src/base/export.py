# coding: utf-8
from __future__ import print_function
import tensorflow as tf
import time
from PIL import Image
import src.base.model as model
from src.web.models import Filter
from src.web.models import FilterResult
import base64
import os
import random
import numpy as np


def get_coco2014_dataset():
    path_dir = os.listdir('res/train2014')
    all_coco2014 = []
    for file in path_dir:
        all_coco2014.append('res/train2014/'+file)
    print(len(all_coco2014))
    image_list = []
    for i in range(10):
        image_list.append(all_coco2014[random.randint(0, len(all_coco2014)-1)])
    return image_list


def center_crop(image, x, y):
    width, height = image.size[0], image.size[1]
    crop_side = min(width, height)
    width_crop = (width-crop_side)//2
    height_crop = (height-crop_side)//2
    box = (width_crop, height_crop, width_crop+crop_side, height_crop+crop_side)
    image = image.crop(box)
    image = image.resize((x, y), Image.ANTIALIAS)
    return image


def export(ckpt_file, model_name, filter_info: Filter):
    g = tf.Graph()      # A new graph
    with g.as_default():
        with tf.Session() as sess:

            processed_image = tf.placeholder(tf.float32, [1, None, None, 3], name='input')
            generated_image = model.net(processed_image, training=False)
            casted_image = tf.cast(generated_image, tf.int32)
            # Remove batch dimension
            squeezed_image = tf.squeeze(casted_image, [0])
            stylized_image_data = tf.reshape(squeezed_image, [-1], name='output')  # 有用的
            # Restore model variables.
            saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V1)
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

            saver.restore(sess, ckpt_file)

            coco2014_list = get_coco2014_dataset()
            filter_result = FilterResult()
            for idx, filename in enumerate(coco2014_list):
                image = Image.open(filename)
                image = image.convert('RGB')
                image_input = center_crop(image, 512, 512)
                start_time = time.time()
                image_output = sess.run(tf.image.encode_jpeg(tf.cast(squeezed_image, tf.uint8)), feed_dict={
                    processed_image: [np.array(image_input)]
                })
                setattr(filter_result, "image"+str(idx+1), base64.urlsafe_b64encode(image_output).decode())
                end_time = time.time()
                tf.logging.info('Elapsed time: %fs' % (end_time - start_time))
            filter_result.save()
            filter_ = Filter.objects.filter(id=filter_info.id)[0]
            filter_.result = filter_result
            filter_.save()

            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, sess.graph_def, output_node_names=['output'])

            with tf.gfile.FastGFile('res/export/'+str(model_name)+'.pb', mode='wb') as f:
                f.write(output_graph_def.SerializeToString())
