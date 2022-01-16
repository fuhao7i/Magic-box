'''
    author: fuhao7i
    2021.01.16
    "用于语义分割标签文件的处理，将RGB格式的label图像转换为单通道的灰度图像，并且像素值0为背景，像素值1为...， 以此类推"        
'''

# 获取label图像中所有种类对应的像素值
import os
from PIL import Image

path = '/content/drive/MyDrive/水下语义分割/datasets/UWSS/labels/'

filenames = os.listdir(path)

classes = []
for fn in filenames:

    img = Image.open(path + fn)
    (w, h) = img.size
    L = img.convert('L')
    for i in range(w):
        for j in range(h):

            if L.getpixel((i,j)) not in classes:
                print(L.getpixel((i,j)))
                classes.append(L.getpixel((i,j)))

# 先运行上面的程序，获得灰度图中每一类对应的像素值，之后依据获得的像素值，将label图像转换为可以用于分割的格式；
import os
from PIL import Image

path = '/content/drive/MyDrive/水下语义分割/datasets/UWSS/labels/'

filenames = os.listdir(path)

for fn in filenames:

    img = Image.open(path + fn)
    (w, h) = img.size
    L = img.convert('L')
    for i in range(w):
        for j in range(h):

            if L.getpixel((i,j)) == 158:  # 海胆
                L.putpixel((i,j), 1)
            elif L.getpixel((i,j)) == 29: # 海参
                L.putpixel((i,j), 2)
            else:
                L.putpixel((i,j), 0)
                
    if fn.endswith('jpg'):
        fn.replace("jpg", "png")
    L.save('/content/drive/MyDrive/水下语义分割/datasets/UWSS/anno/' + fn)

print('==> 转换完成！')