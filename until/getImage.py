import requests
from PIL import Image
import io
import numpy as np

url = "http://jw1.wucc.cn/CheckCode.aspx"


def getImage():
    for i in range(1, 800):
        try:
            image_content = requests.get(url)
        except Exception as e:
            print(e)
        else:
            with open("Images/" + str(i) + ".png", 'wb') as f:
                f.write(image_content.content)
            print(i, "下载完成!")

res = requests.get(url)
image_file = io.BytesIO(res.content)
image = Image.open(image_file)
a=np.concatenate(np.array(image))
unique,counts=np.unique(a,return_counts=True)
#计算像素值
print(sorted(list(zip(counts,unique)),reverse=True)[:10])