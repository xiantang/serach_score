from PIL import Image
from requests import session
import numpy as  np
import tensorflow as tf
from keras.models import load_model
import io
from keras.backend import image_data_format
import re
from bs4 import BeautifulSoup

from requests.utils import quote

img_rows, img_cols = 12, 22

if image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)

import string

CHRS = string.ascii_lowercase + string.digits

model = load_model(r'ok.h5')
graph = tf.get_default_graph()

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
def Login(studentId,password):
    while True:
        S=session()
        images=S.get("http://jw1.wucc.cn/CheckCode.aspx").content
        image_file = io.BytesIO(images)
        images = Image.open(image_file)
        # images.show()
        images=handle_split_image(images)
        images=_predict_image(images)
        login_data = {
            '__VIEWSTATE': 'dDwyODE2NTM0OTg7Oz5C03XAlD/5uyAktVelR/+o3PpAIQ==',
            'txtUserName': studentId,
            'TextBox2': password,
            'txtSecretCode': images,
            'RadioButtonList1': '学生'.encode('gbk'),  # 表示学生登录
            'Button1': '',
            'lbLanguage': '',
            'hidPdrs': '',
            'hidsc':  '',
        }
        text=S.post("http://jw1.wucc.cn/default2.aspx",data=login_data).text

        username = re.findall('xhxm">(.*?)</span>', text)

        if "错误" in text:
            return "密码错误"
        elif username !=[] and "同学" in username[0]:
            return S,username,studentId
        elif "用户名不存在" in text:
            return  "用户名不存在"


def get_score(S, username,studentId):
    new_referer = {'referer': 'http://jw1.wucc.cn/xs_main.aspx?xh=' + studentId[0]}
    S.headers.update(new_referer)
    username = username[0].replace("同学", "")
    url_history_grade = "http://jw1.wucc.cn/" + (
        'xscjcx.aspx?'
        'xh={0}'
        '&xm={1}'
        '&gnmkdm=N121605'
    ).format(studentId, quote(username.encode('gbk')))
    text = S.get(url_history_grade).text
    soup = BeautifulSoup(text, "lxml")
    csrf_token = soup.find('input', attrs={'name': '__VIEWSTATE'})['value']
    data = {
        '__EVENTTARGET': '',
        '__EVENTARGUMENT': '',
        'hidLanguage': '',
        '__VIEWSTATE': csrf_token,
        'ddLXN': '',
        'ddLXQ': '',
        'ddl_kcxz': '',
        'btn_zcj': '',  # 经试验该项可为空 # '历年成绩', # ,'%EF%BF%BD%EF%BF%BD%EF%BF%BD%EF%BF%BD%C9%BC%EF%BF%BD'
    }

    res = S.post(url_history_grade, data=data)

    soup = BeautifulSoup(res.text, "lxml")
    table = soup.select_one('.datelist')
    keys = [i.text for i in table.find('tr').find_all('td')]
    scores = [
        dict(zip(
            keys, [i.text.strip() for i in tr.find_all('td')]))
        for tr in table.find_all('tr')[1:]]
    return  scores

def getScore(studentId,passWord):
    data = Login(studentId, passWord)
    if data == "密码错误":
        return "密码错误"
    elif data== "用户名不存在":
        return "用户名不存在"
    else:
        S,username,studentId=data
        score = get_score(S, username,studentId)
        return score


