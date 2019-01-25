import requests
import base64
import json
from io import BytesIO
from PIL import Image


def train_test():
    with open("res/image/mosaic.jpg", "rb") as f:
        base64_data = base64.urlsafe_b64encode(f.read())
        base64_data = base64_data.decode()
    print(base64_data)
    filter_info = json.dumps({
        'upload_id': 255, # 唯一标识，不能重复
        'filter_name': 'test', # 滤镜名称
        'owner': 'czczcz',
        'style_template': base64_data,
        'brush_size': 768, # 笔刷大小，无需修改
        'brush_intensity': 1000, # 风格强度，可以修改
        'smooth': 1000
    })
    response = requests.post('http://120.79.162.134:7005/api/createFilter', data=filter_info)
    print(response)


def image_test():
    with open("res/image/mosaic.jpg", "rb") as f:
        base64_data = base64.urlsafe_b64encode(f.read())
        base64_data = base64_data.decode()
    image = BytesIO(base64.urlsafe_b64decode(base64_data))
    image = Image.open(image)
    width = image.size[0]
    height = image.size[1]
    scale = 80 / height
    image = image.resize((int(width*scale), int(height*scale)), Image.ANTIALIAS)
    image.show()


def get_coco2014_dataset():
    import os
    import random
    path_dir = os.listdir('res/train2014')
    all_coco2014 = []
    for file in path_dir:
        all_coco2014.append('res/train2014/'+file)
    print(len(all_coco2014))
    image_list = []
    for i in range(1000):
        image_list.append(all_coco2014[random.randint(0, len(all_coco2014)-1)])
    return image_list


if __name__ == '__main__':
    train_test()
    # image_test()
    # get_coco2014_dataset()