# coding: utf-8
from __future__ import print_function
from __future__ import division
import tensorflow as tf
import src.base.reader as reader
import src.base.model as model
from src.base.nets import nets_factory
from src.base.preprocessing import preprocessing_factory
from src.base.export import export
from src.web.models import Filter
import time
import src.base.losses as losses
import src.base.utils as utils
import os
import zipfile
import shutil


class Flags(object):
    iter_num = 8000
    batch_size = 2
    epoch = 1
    content_weight = 1
    content_layers = ["vgg_16/conv3/conv3_3"]
    style_layers = ["vgg_16/conv1/conv1_2", "vgg_16/conv2/conv2_2",
                    "vgg_16/conv3/conv3_3", "vgg_16/conv4/conv4_3"]
    model_path = 'res/models'
    loss_model = "vgg_16"
    checkpoint_exclude_scopes = "vgg_16/fc"  # we only use the convolution layers, so ignore fc layers.
    loss_model_file = "res/pretrained/vgg_16.ckpt"  # the path to the checkpoint
    target = 'res/target/'
    dataset = 'res/train2014'


class Loss(object):
    def __init__(self, content_loss, style_loss, style_loss_summary, tv_loss, loss):
        self.content_loss = content_loss
        self.style_loss = style_loss
        self.style_loss_summary = style_loss_summary
        self.tv_loss = tv_loss
        self.loss = loss


def init_network(filter_info: Filter, debug):
    """
    初始化神经网络
    :param filter_info: 待训练的模型的参数
    :param debug: 是否要debug
    :return: style_features_t, training_path
    """

    style_features_t = losses.get_style_features(filter_info, Flags, debug)
    # Make sure the training path exists.
    training_path = os.path.join(Flags.model_path, str(filter_info.id))
    return style_features_t, training_path


def build_network(filter_info: Filter):
    """
    建立神经网络
    :param filter_info: 待训练的模型的参数
    :return: generated, endpoints_dict, processed_images
    """
    network_fn = nets_factory.get_network_fn(num_classes=1, is_training=False)
    global image_preprocessing_fn, image_unprocessing_fn
    image_preprocessing_fn, image_unprocessing_fn = preprocessing_factory.get_preprocessing(is_training=False)
    processed_images = reader.image(Flags.batch_size, filter_info.brush_size, filter_info.brush_size,
                                    Flags.dataset, image_preprocessing_fn, epochs=Flags.epoch)
    # print(processed_images)
    generated = model.net(processed_images, training=True)
    processed_generated = [image_preprocessing_fn(image, filter_info.brush_size, filter_info.brush_size)
                           for image in tf.unstack(generated, axis=0, num=Flags.batch_size)]
    processed_generated = tf.stack(processed_generated)
    _, endpoints_dict = network_fn(tf.concat([processed_generated, processed_images], 0), spatial_squeeze=False)
    return generated, endpoints_dict, processed_images


def build_losses(filter_info: Filter, endpoints_dict, style_features_t, generated):
    """
    计算模型损失
    :param filter_info: 待训练的模型的参数
    :param endpoints_dict:
    :param style_features_t: VGG网络计算得到的特征
    :param generated: 生成图像
    :return:
    """
    content_loss = losses.content_loss(endpoints_dict, Flags.content_layers)
    style_loss, style_loss_summary = losses.style_loss(endpoints_dict, style_features_t, Flags.style_layers)
    tv_loss = losses.total_variation_loss(generated)  # use the unprocessed image
    loss = filter_info.brush_intensity*style_loss+Flags.content_weight*content_loss+filter_info.smooth*tv_loss
    return Loss(content_loss, style_loss, style_loss_summary, tv_loss, loss)


def add_summary(filter_info: Filter, loss: Loss, processed_images, generated, training_path):
    """
    在TensorBoard中记录训练数据
    :param filter_info: 待训练的模型的参数
    :param loss: 损失对象
    :param processed_images: 原图
    :param generated: 风格化图像
    :param training_path: 模型保存路径
    :return: summary, writer
    """
    tf.summary.scalar('losses/content_loss', loss.content_loss)
    tf.summary.scalar('losses/style_loss', loss.style_loss)
    tf.summary.scalar('losses/regularizer_loss', loss.tv_loss)
    tf.summary.scalar('weighted_losses/weighted_content_loss', loss.content_loss*Flags.content_weight)
    tf.summary.scalar('weighted_losses/weighted_style_loss', loss.style_loss*filter_info.brush_intensity)
    tf.summary.scalar('weighted_losses/weighted_regularizer_loss', loss.tv_loss*filter_info.smooth)
    tf.summary.scalar('total_loss', loss.loss)

    for layer in Flags.style_layers:
        tf.summary.scalar('style_losses/'+layer, loss.style_loss_summary[layer])
    origin = tf.stack(
        [image_unprocessing_fn(image) for image in tf.unstack(processed_images, axis=0, num=Flags.batch_size)])
    tf.summary.image('generated', generated)
    tf.summary.image('origin', origin)
    list_generated = [image for image in tf.unstack(generated, axis=0, num=Flags.batch_size)]
    list_origin = [image for image in tf.unstack(origin, axis=0, num=Flags.batch_size)]
    list_color_origin = []
    for cnt in range(len(list_generated)):
        h, s, v = tf.split(tf.image.rgb_to_hsv(list_generated[cnt]), 3, axis=2)
        hh, ss, vv = tf.split(tf.image.rgb_to_hsv(list_origin[cnt]), 3, axis=2)
        hhssv = tf.concat([hh, ss, v], 2)
        list_color_origin.append(tf.image.hsv_to_rgb(hhssv))
    color_origin = tf.stack(list_color_origin)
    tf.summary.image('color_origin', color_origin)
    summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(training_path)
    return summary, writer


def prepare_train(sess: tf.Session, loss: Loss, training_path):
    """
    准备训练
    :param sess: TensorFlow计算图
    :param loss: Loss对象
    :param training_path: 模型保存路径
    :return:
    """
    global_step = tf.Variable(0, name="global_step", trainable=False)

    variable_to_train = []
    for variable in tf.trainable_variables():
        if not (variable.name.startswith(Flags.loss_model)):
            variable_to_train.append(variable)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss.loss, global_step=global_step, var_list=variable_to_train)

    variables_to_restore = []
    for v in tf.global_variables():
        if not (v.name.startswith(Flags.loss_model)):
            variables_to_restore.append(v)
    variables_to_restore = [var for var in variables_to_restore if 'Adam' not in var.name]
    saver = tf.train.Saver(variables_to_restore, max_to_keep=1)

    sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    # Restore variables for loss network.
    init_func = utils._get_init_fn(Flags)
    init_func(sess)

    # Restore variables for training model if the checkpoint file exists.
    last_file = tf.train.latest_checkpoint(training_path)
    if last_file:
        saver.restore(sess, last_file)
    # else:
    #     print("正在载入base model")
    #     saver.restore(sess, "res/baseModel/base.ckpt")
    return train_op, saver, global_step


def start_train(sess, saver, loss, train_op, global_step, summary, writer, training_path, debug, filter_info: Filter):
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    start_time = time.time()
    try:
        while not coord.should_stop():
            _, loss_t, step = sess.run([train_op, loss.loss, global_step])
            elapsed_time = time.time()-start_time
            start_time = time.time()
            # logging
            print('step: %d,  total Loss %f, secs/step: %f'%(step, loss_t, elapsed_time))
            if step % 50 == 0:
                filter = Filter.objects.filter(id=filter_info.id)[0]
                filter.schedule = step / Flags.iter_num * 100
                filter.save()
            # summary
            if debug:
                if step % 50 == 0:
                    summary_str = sess.run(summary)
                    writer.add_summary(summary_str, step)
                    writer.flush()
            # checkpoint
            if step % 500 == 0:
                if not (os.path.exists(training_path)):
                    os.makedirs(training_path)
                saver.save(sess, os.path.join(training_path, 'model.ckpt'), global_step=step)
            if step >= Flags.iter_num:
                break
    finally:
        coord.request_stop()
    coord.join(threads)


def get_zip(pb_file, zip_file, ckpt_path, debug=False):
    azip = zipfile.ZipFile(zip_file, 'w')
    azip.write(pb_file, compress_type=zipfile.ZIP_LZMA)
    azip.close()
    os.remove(pb_file)
    if not debug:
        shutil.rmtree(ckpt_path)


def train(filter_info: Filter, debug):
    """
    训练模型
    :param filter_info: 待训练的模型的参数
    :param debug: 是否debug
    :return:
    """
    style_features_t, training_path = init_network(filter_info, debug)
    with tf.Graph().as_default():
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.99
        with tf.Session(config=tf_config) as sess:
            generated, endpoints_dict, processed_images = build_network(filter_info)
            loss_tensor = build_losses(filter_info, endpoints_dict, style_features_t, generated)
            if debug:
                summary, writer = add_summary(filter_info, loss_tensor, processed_images, generated, training_path)
            else:
                summary, writer = None, None
            train_op, saver, global_step = prepare_train(sess, loss_tensor, training_path)
            start_train(sess, saver, loss_tensor, train_op, global_step, summary,
                        writer, training_path, debug, filter_info)
            export(ckpt_file=training_path+'/model.ckpt-'+str(Flags.iter_num), model_name=filter_info.upload_id, filter_info=filter_info)
            get_zip(pb_file='res/export/'+str(filter_info.upload_id)+'.pb',
                    zip_file='res/export/'+str(filter_info.upload_id)+'.zip',
                    ckpt_path=training_path, debug=debug)
