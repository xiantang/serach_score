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

