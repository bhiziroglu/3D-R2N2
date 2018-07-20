'''Reads and parses training data and labels'''
import os
import os.path
import binvox_rw
import numpy as np
from PIL import Image
import tensorflow as tf

TEST_DATA = "03001627" #03001627 -> 13556 OBJE, 176228 RESIM TOPLAMDA
MAIN_DATA = "03211117"

BATCH_SIZE = 2
label_dir = os.listdir('./'+TEST_DATA+'_labels')
data = os.listdir('./'+TEST_DATA+'')
TOTAL_SIZE = len(data)


def train_labels():
    
    if(len(label_dir)<BATCH_SIZE):
        return []

    y_train = {} 
    #d = {} # Keys = IDs for items
           # Values = binvox
    for _ in range(BATCH_SIZE):
        label = label_dir.pop()
        if(label.startswith('.')):
            continue
        binv = open('./'+TEST_DATA+'_labels/'+label+'/model.binvox','rb')
        binvox_data = binvox_rw.read_as_3d_array(binv).data # binvox_data is 32x32x32
        y_train[label] = np.asarray(binvox_data)

    return y_train
        


def train_data():
    
    if(len(data)<BATCH_SIZE):
        return []


    x_train = {} #Keys = IDs for items, Values = 24 pictures

    for _ in range(BATCH_SIZE):
        item = data.pop()
        tmp_im_array = []
        if(item.startswith('.')):
            continue
        for pic in os.listdir('./'+TEST_DATA+'/'+item+'/rendering'):
            if('.png' in pic): #If current item is a picture, store it
                im = np.array(Image.open('./'+TEST_DATA+'/'+item+'/rendering/'+pic))
                im = tf.random_crop(im,[127,127,3]) # Input images are [137,137,3]. Remove random 10x10 pixel area.
                tmp_im_array.append(im)
                
        x_train[item] = tmp_im_array
        tmp_im_array = []

    return x_train