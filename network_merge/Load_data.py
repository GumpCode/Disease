#coding: utf-8

import os
from PIL import Image
import numpy as np

#读取文件夹mnist下的42000张图片，图片为灰度图，所以为1通道，
#如果是将彩色图作为输入,则将1替换为3，并且data[i,:,:,:] = arr改为data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
def load_data():
    data = np.empty((793,3,64,64),dtype="float32")
    label = np.empty((793,),dtype="uint8")

    imgs = os.listdir("/home/ganlinhao/DeepLearning/keras/Disease/network_merge/data")
    num = len(imgs)
    for i in range(num):
        img = Image.open("/home/ganlinhao/DeepLearning/keras/Disease/network_merge/data/"+imgs[i])
        arr = np.asarray(img,dtype="float32")
        data[i,:,:,:] = [arr[:,:,0], arr[:,:,1], arr[:,:,2]]
        label[i] = int(imgs[i].split('_')[0])
    return data, label
