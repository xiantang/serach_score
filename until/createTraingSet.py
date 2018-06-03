from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageFilter, ImageEnhance

import random
import string
CHRS = string.ascii_lowercase + string.digits  # 小写字母+数字的字符串列表

t_size = (12, 22)
font_size = 20 # 即字体的高(大概值), 宽度约为一半
font_path = [  # 字体路径

    r'C:/Windows/Fonts/VERDANA.TTF',
    r'C:/Windows/Fonts/SIMKAI.TTF',
    r'C:/Windows/Fonts/SIMSUNB.TTF',
    r'C:/Windows/Fonts/REFSAN.TTF',
    r'C:/Windows/Fonts/MSYH.TTC',
]


fonts = [ImageFont.truetype(fp, font_size) for fp in font_path]
# font = ImageFont.truetype('E:/python/2017_9/simhei.ttf', font_size)

def gen_fake_code(prefix, c, font):
    txt = Image.new('L', t_size, color='black')
    ImageDraw.Draw(txt).text((0, -2), c, fill='white', font=font)
    w = txt.rotate(random.uniform(-20, 20), Image.BILINEAR) # center不指定，默认为中心点
    img_ = w.point(lambda i: i < 10, mode='1')
    # img_.show()
    img_.save(prefix + '_' + c + '.png')

# if __name__ == '__main__':
import os
os.chdir(r'images')
for c in CHRS:  # 对每一个字符的每一种字体生成200张
    for n, font in enumerate(fonts):
        for i in range(200):
            gen_fake_code(str(n)+'-'+str(i), c, font)

