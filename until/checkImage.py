import requests
import io
import numpy as  np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
from keras.backend import image_data_format

img_rows, img_cols = 12, 22

if image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)

import string

CHRS = string.ascii_lowercase + string.digits

model = load_model(r'ok.h5')
graph = tf.get_default_graph()


def get_image():
    '''
    从教务处网站获取验证码
    '''

    # url_base = requests.get('http://xsweb.scuteo.com/default2.aspx').url.replace('default2.aspx', '')
    url_veri_img = 'http://jw1.wucc.cn/CheckCode.aspx'
    headers = {
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36',
    }
    try:
        res = requests.get(url_veri_img, headers=headers,timeout=3)
    except Exception as e:
        print(e)
    else:
        image_file = io.BytesIO(res.content)
        image = Image.open(image_file)
        import time
        time.sleep(2)
        return image


def handle_split_image(image):
    '''
    切割验证码，返回包含四个字符图像的列表
    '''
    im = image.point(lambda i: i != 43, mode='1')
    y_min, y_max = 0, 22  # im.height - 1 # 26
    split_lines = [5, 17, 29, 41, 53]
    ims = [im.crop([u, y_min, v, y_max]) for u, v in zip(split_lines[:-1], split_lines[1:])]
    # w = w.crop(w.getbbox()) # 切掉白边 # 暂不需要
    return ims


def _predict_image(images):
    global graph
    Y = []
    for i in range(4):
        im = images[i]
        test_input = np.concatenate(np.array(im))
        test_input = test_input.reshape(1, *input_shape)
        y_probs = None
        with graph.as_default():
            y_probs = model.predict(test_input)
        y = CHRS[y_probs[0].argmax(-1)]
        Y.append(y)

    return ''.join(Y)


def multi_process(x):
    '''
    获取预测并保存图片，图片名为预测值
    '''
    image = get_image()
    images = handle_split_image(image)
    image.save( "modeltest/"+_predict_image(images) + '.png')
    print("ok")

from multiprocessing.dummy import Pool

import datetime
now = datetime.datetime.now
start = now()
with Pool(30) as pool:
    pool.map(multi_process, [i for i in range(300)])
print('耗时 -> %s' % (now()-start))



