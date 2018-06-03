###调参侠初体验 

首先获取温州商学院的验证码 
[温州商学院验证码](http://jw1.wucc.cn/CheckCode.aspx)

对验证码进行降噪 并且切割

```
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
```

得到下图

![GitHub set up](https://upload-images.jianshu.io/upload_images/2422746-7f01203ce800b4a3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/538)

下面是构建训练集

```
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from PIL import Image
import numpy as np
import requests
import os
import io


os.chdir(r'images')  # 跳转到训练集目录
import string

CHRS = string.ascii_lowercase + string.digits  # 字符列表

num_classes = 36  # 共要识别36个字符（所有小写字母+数字），即36类
batch_size = 128
epochs = 12

# 输入图片的尺寸
img_rows, img_cols = 12, 22
# 根据keras的后端是TensorFlow还是Theano转换输入形式
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)

import glob

X, Y = [], []
for f in glob.glob('*.png')[:]:  # 遍历当前目录下所有png后缀的图片
    t = 1.0 * np.array(Image.open(f))
    t = t.reshape(*input_shape)  # reshape后要赋值
    X.append(t)  # 验证码像素列表

    s = f.split('_')[1].split('.')[0]  # 获取文件名中的验证码字符
    Y.append(CHRS.index(s))  # 将字符转换为相应的0-35数值

X = np.stack(X)  # 将列表转换为矩阵
Y = np.stack(Y)
# 此时Y形式为 array([26, 27, 28, ..., 23, 24, 25])

# 对Y值进行one-hot编码 # 可尝试 keras.utils.to_categorical(np.array([0,1,1]), 3) 理解
Y = keras.utils.to_categorical(Y, num_classes)

split_point = len(Y) - 720  # 简单地分割训练集与测试集
x_train, y_train, x_test, y_test = X[:split_point], Y[:split_point], X[split_point:], Y[split_point:]

# 以下模型和mnist-cnn相同
# 两层3x3窗口的卷积(卷积核数为32和64)，一层最大池化(MaxPooling2D)
# 再Dropout(随机屏蔽部分神经元)并一维化(Flatten)到128个单元的全连接层(Dense)，最后Dropout输出到36个单元的全连接层（全部字符为36个）

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```
主要原理是将windows 自带的字体分别对验证码的字母数据
进行随机度数的旋转 用来模仿验证码的旋转度数，训练之后有20
%的识别概率

下面是使用python 自带的Tk库进行人工识别 并把识别完成的验证码
放入new文件夹中

```
import os
import glob

image_files = glob.glob('*.png')

from tkinter import *

root = Tk()

colours = ['green', 'orange', 'white'] * 2
labels = []
entrys = []
strvars = []

for r, c in enumerate(colours):
    l = Label(root, text=c, relief=RIDGE, width=34)
    l.grid(row=r, column=0)
    labels.append(l)

    v = StringVar(root, value='predict')
    strvars.append(v)

    e = Entry(root, textvariable=v, bg=c, relief=SUNKEN, width=10,
              font="Helvetica 44 bold")
    e.grid(row=r, column=1)
    entrys.append(e)

info_label1 = Label(root, text='当前正确率', relief=RIDGE, width=34)
info_label1.grid(row=7, column=0)
info_label2 = Label(root, text='已用时间', relief=RIDGE, width=34)
info_label2.grid(row=7, column=1)

# ims = []
num = 0
cur_files = None

correct = 0
incorrect = 0

if not os.path.exists('new'):
    os.mkdir('new')

import datetime

now = datetime.datetime.now
start = now()

from PIL import Image, ImageTk


def enter_callback(e):
    global num, cur_files
    global correct, incorrect

    if cur_files:
        for i in range(6):
            name = strvars[i].get()
            # print(name)

            if cur_files[i].split('.')[0] == name:
                correct += 1
            else:
                incorrect += 1
            try:
                os.rename(cur_files[i], ''.join(['new/', name, '.png']))
            except Exception as e:
                print(e)

        info1 = '当前正确率: %s' % (correct / (correct + incorrect))
        info2 = '已用时间: %s' % (now() - start)
        info_label1.config(text=info1)
        info_label2.config(text=info2)
    else:
        for i in range(6):
            labels[i].config(width=144)

    cur_files = image_files[num: num + 6]

    for i in range(6):
        f = image_files[num + i]
        im = Image.open(f).resize((144, 54))
        im = ImageTk.PhotoImage(im)
        # https://stackoverflow.com/questions/18369936/how-to-open-pil-image-in-tkinter-on-canvas
        # im = PhotoImage(file=f)
        labels[i].configure(image=im)
        labels[i].image = im
        strvars[i].set(f.split('.')[0])
    num += 6


root.bind("<Return>", enter_callback)
root.mainloop()

```
![GitHub set up](https://upload-images.jianshu.io/upload_images/2422746-339e378f3f81db66.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/658)

识别到200张左右就可以把之前的模型 继续训练 

```
import glob
import keras
from keras import backend as K
from PIL import Image
import numpy as np
import os
import string
from keras.models import load_model
num_classes = 36
batch_size = 128
# 输入图片的尺寸
img_rows, img_cols = 12, 22
CHRS = string.ascii_lowercase + string.digits  # 字符列表
model = load_model(r'ok.h5')
os.chdir(r'C:\Users\战神皮皮迪\Desktop\login_school\new')
# 检查是否有打错码如打成3个字符的图片
file_names = glob.glob('*.png')
epochs = 12
print(file_names)

def handle_split_image(image):
    '''
    切割验证码，返回包含四个字符图像的列表
    '''
    im = image.point(lambda i: i != 43, mode='1')
    y_min, y_max = 0, 22 # im.height - 1 # 26
    split_lines = [5,17,29,41,53]
    ims = [im.crop([u, y_min, v, y_max]) for u, v in zip(split_lines[:-1], split_lines[1:])]
    # w = w.crop(w.getbbox()) # 切掉白边 # 暂不需要
    return ims

error = [i for i in file_names if len(i.split('.')[0])!=4]
if error:
    print(error)
    raise Exception('打码出错，请检查')
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)
# 构造训练集
X, Y = [], []
for f in file_names:
    image = Image.open(f)
    ims = handle_split_image(image)  # 打开并切割图片

    name = f.split('.')[0]
    # 将图片切割出的四个字符及其准确值依次放入列表
    for i, im in enumerate(ims):
# 以下类同上文
        t = 1.0 * np.array(im)
        t = t.reshape(*input_shape)
        X.append(t)

        s = name[i]
        Y.append(CHRS.index(s))  # 验证码字符

X = np.stack(X)
Y = np.stack(Y)

Y = keras.utils.to_categorical(Y, num_classes)

split_point = len(Y) - 1000
x_train, y_train, x_test, y_test = X[:split_point], Y[:split_point], X[split_point:], Y[split_point:]

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
```
得到的验证码能够到40%的正确率  
然后我们使用这个模型去采集并且识别验证码 
然后继续人工识别  最后能将正确率提高到95%

```
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


```
