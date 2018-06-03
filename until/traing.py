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


# model.save("model_test.h5")
os.chdir(r'C:\Users\战神皮皮迪\Desktop\login_school\modeltest\new')
# 检查是否有打错码如打成3个字符的图片
file_names = glob.glob('*.png')
error = [i for i in file_names if len(i.split('.')[0])!=4]
if error:
    print(error)
    raise Exception('打码出错，请检查')

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

model.save(r'C:\Users\战神皮皮迪\Desktop\login_school\ok.h5')
