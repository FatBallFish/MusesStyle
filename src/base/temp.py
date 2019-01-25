import tensorflow as tf
import model
from PIL import Image
import time
import numpy as np
import sys

image_list = [
    "../../res/image/test1.jpg",
    "../../res/image/test2.jpg",
    "../../res/image/test3.jpg",
    "../../res/image/test4.jpg",
]

def fill_image(image):
    width, height = image.size
    print(width, height)
    new_image_length = width if width > height else height
    print(new_image_length)
    new_image = Image.new(image.mode, (new_image_length, new_image_length), color='white')
    if width > height:
        new_image.paste(image, (0, int((new_image_length - height) / 2)))
    else:
        new_image.paste(image, (int((new_image_length - width) / 2), 0))
    return new_image


def cut_image(image):
    width, height = image.size
    item_width = int(width / 3)
    box_list = []
    count = 0
    for j in range(0, 3):
        for i in range(0, 3):
            count += 1
            box = (i * item_width, j * item_width, (i + 1) * item_width, (j + 1) * item_width)
            box_list.append(box)
    print(count)
    image_list = [image.crop(box) for box in box_list]
    return image_list


def merge_image(image_list, name):
    image_list = [image_list[i:i+3] for i in range(0, len(image_list), 3)]
    width, height = image_list[0].size
    target = Image.new('RGB', (width * 3, width * 3))
    for i, row in enumerate(image_list):
        for j, item in enumerate(row):
            a = j * width  # 图片距离左边的大小
            b = i * width  # 图片距离上边的大小
            c = a + width # 图片距离左边的大小 + 图片自身宽度
            d = b + width  # 图片距离上边的大小 + 图片自身高度
            target.paste(item, (a, b, c, d))
    target.save("../../res/test/"+name+".png")


def save_images(image_list):
    index = 1
    for image in image_list:
        image.save('result/' + str(index) + '.png')
        index += 1


if __name__ == '__main__':
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
            saver.restore(sess, "../../res/models/255/model.ckpt-17000")

            image_list = []
            for idx, filename in enumerate(image_list):
                image = Image.open(filename)
                image = fill_image(image)
                image_list = cut_image(image)
                image_result = []
                for j, image in enumerate(image_list):
                    start_time = time.time()
                    image_output = sess.run(tf.cast(squeezed_image, tf.uint8), feed_dict={
                        processed_image: [np.array(image)]
                    })
                    end_time = time.time()
                    tf.logging.info('Elapsed time: %fs' % (end_time - start_time))
                    im = Image.fromarray(image_output)
                    image_result.append(im)
                else:
                    merge_image(image_result, str(idx))
