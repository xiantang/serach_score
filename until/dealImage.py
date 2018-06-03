from PIL import Image, ImageFilter, ImageEnhance
from collections import Counter
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def plan1(image):
    """
    灰度化,二值化,中值滤波,锐化
    :param image:
    :return:
    """
    imageL = image.convert("L").point(
        lambda i: i > 25, mode="1").filter(
        ImageFilter.MedianFilter(3)).convert('L')
    ImageS = ImageEnhance.Sharpness(imageL).enhance(2.0)
    return ImageS


def cutLine1(image):
    a = np.array(image)
    pd.DataFrame(a.sum(axis=0)).plot.line()
    plt.imshow(a)
    split_lines=[5,17,29,41,53]
    vlines=[plt.axvline(i,color='r') for i in split_lines]
    plt.show()



image = Image.open("Images/1.png")  # 打开图片
image = plan1(image)

cutLine1(image)
