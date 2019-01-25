import tensorflow as tf
import model
from PIL import Image
import time
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
image_list = [
    "../../res/image/test1.jpg",
    # "../../res/image/test2.jpg",
    # "../../res/image/test3.jpg",
    # "../../res/image/test4.jpg",
]

class ImageItem:
    def __init__(self, id, image):
        self.id = id
        self.image = image


def fill_image(image):
    width, height = image.size
    print(width, height)
    new_image_length = width if width > height else height
    # new_image_length = (width // 30 + 1) * 30
    print(new_image_length)
    new_image = Image.new(image.mode, (new_image_length, new_image_length), color='white')
    if width > height:
        new_image.paste(image, (0, int((new_image_length - height) / 2)))
    else:
        new_image.paste(image, (int((new_image_length - width) / 2), 0))
    return new_image

def padding(image, padding):
    width, height = image.size
    new_image = Image.new(image.mode, (width+2*padding, height+2*padding), color='white')
    new_image.paste(image, (padding, padding, width+padding, height+padding))
    return new_image

def unpadding(image, padding):
    width, height = image.size
    box = (padding, padding, width-padding, height-padding)
    image = image.crop(box)
    return image

def cut_image(image, part, padding):
    width, height = image.size
    item_width = int((width-2*padding) / part)
    box_list = []
    count = 0
    for j in range(0, part):
        for i in range(0, part):
            count += 1
            box = (i * item_width, j * item_width,
                   (i + 1) * item_width+2*padding, (j + 1) * item_width+2*padding)
            box_list.append(box)
    print(count)
    image_list = [image.crop(box) for box in box_list]
    return image_list


def merge_image(image_list, part):
    image_list = [image_list[i:i+part] for i in range(0, len(image_list), part)]
    width, height = image_list[0][0].size
    print(width)
    print("图像的像素为: ",width * part,"*",width * part)
    target = Image.new('RGB', (width * part, width * part))
    for i, row in enumerate(image_list):
        for j, item in enumerate(row):
            a = j * width  # 图片距离左边的大小
            b = i * width  # 图片距离上边的大小
            c = a + width # 图片距离左边的大小 + 图片自身宽度
            d = b + width  # 图片距离上边的大小 + 图片自身高度
            target.paste(item, (a, b, c, d))
    return target


def optimize_seam(image_list, part, seam):
    image_list = [image_list[i:i+part] for i in range(0, len(image_list), part)]
    width, height = [item - 2*seam for item in image_list[0][0].size]
    width_padding, height_padding = image_list[0][0].size
    print(width)
    print("图像的像素为: ",width * part,"*",width * part)
    target = Image.new('RGB', (width*part+2*seam, width*part+2*seam))
    for i, row in enumerate(image_list):
        for j, item in enumerate(row):
            a = j * width + seam  # 图片距离左边的大小
            b = i * width + seam  # 图片距离上边的大小
            c = a + width - 2*seam  # 图片距离左边的大小 + 图片自身宽度
            d = b + width - 2*seam  # 图片距离上边的大小 + 图片自身高度
            print((a,b,c,d, width_padding))
            image_unpadding = unpadding(item, padding=2*seam)
            print(image_unpadding.size)
            target.paste(image_unpadding, (a, b, c, d))
            if j != 0:
                start_x = a - 2 * seam
                end_x = a
                start_y = b - seam
                end_y = d + seam
                for x in range(start_x, end_x):
                    for y in range(start_y, end_y):
                        alpha = (2*seam-(x-start_x))/(2*seam)
                        pixel1 = image_list[i][j-1].getpixel(((width_padding-2*seam-1)+(x-start_x), y-start_y+seam))
                        pixel2 = image_list[i][j].getpixel((x-start_x,y-start_y+seam))
                        pixel_add = tuple([int(pixel1[i] * alpha + pixel2[i] * (1-alpha)) for i in range(3)])
                        target.putpixel((x, y), pixel_add)
                    else:
                        print("x:%d" % x)
            if i != 0:
                start_x = a - seam
                end_x = c + seam
                start_y = b - 2 * seam
                end_y = b
                for y in range(start_y, end_y):
                    for x in range(start_x, end_x):
                        alpha = (2 * seam - (y - start_y)) / (2 * seam)
                        pixel1 = image_list[i-1][j].getpixel(
                            (x - start_x + seam, (width_padding - 2 * seam - 1) + (y - start_y)))
                        pixel2 = image_list[i][j].getpixel((x - start_x + seam, y - start_y))
                        pixel_add = tuple([int(pixel1[i] * alpha + pixel2[i] * (1 - alpha)) for i in range(3)])
                        target.putpixel((x, y), pixel_add)
                    else:
                        print("x:%d" % x)
    return target


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
            saver.restore(sess, "../../res/models/252/model.ckpt-17000")
            pad = 20
            part = 60
            seam = 5
            for idx, filename in enumerate(image_list):
                image = Image.open(filename)
                image = fill_image(image)
                image = padding(image, padding=pad)
                print("第一步：切割成块")
                image_list = cut_image(image, part=part, padding=pad)
                # test = merge_image(image_list, part=part)
                # test.save("../../res/test/test-step1.jpg")
                print("第二步：转化为对象")
                item_list = [ImageItem(id, image) for id, image in enumerate(image_list)]
                print("第三步：打乱顺序")
                np.random.shuffle(item_list) # 随机打乱
                print("第四步：合并图像")
                image = merge_image([item.image for item in item_list], part=part) # 合并

                image = padding(image, padding=pad)
                image_list = cut_image(image, part=3, padding=pad) # 切割成3*3
                print("第五步：切割成3*3块")
                temp_list = []
                temp_list_origin = []

                for j, image in enumerate(image_list):
                    print("正在风格化第%d/9块"%(j+1))
                    start_time = time.time()
                    image_output = sess.run(tf.cast(squeezed_image, tf.uint8), feed_dict={
                        processed_image: [np.array(image)]
                    })

                    end_time = time.time()
                    tf.logging.info('Elapsed time: %fs' % (end_time - start_time))
                    im = Image.fromarray(image_output)
                    temp_list.append(im)
                else:
                    print("第六步：合并图像")
                    temp_list = [unpadding(image, padding=pad) for image in temp_list]
                    image = merge_image(temp_list, part=3)
                    image.save("../../res/test/random%d_test.jpg" % idx)
                    print("第七步：切割成12*12块")
                    image_list = cut_image(image, part=part, padding=0)
                    for index, image in enumerate(image_list):
                        item_list[index].image = unpadding(image, padding=pad-seam) # 用风格化的图像替换原图像
                    print("第八步：恢复原始顺序")
                    item_list = sorted(item_list, key=lambda item: item.id)
                    image_list = [item.image for item in item_list]
                    print("第十步：合并图像")
                    image = optimize_seam(image_list, part=part, seam=seam)
                    image = unpadding(image, padding=20)
                    image.save("../../res/test/random%d_seam.jpg" % idx)

