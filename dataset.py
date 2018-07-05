'''Reads and parses training data and labels'''
import os
import os.path
import binvox_rw
import numpy as np
from PIL import Image
import tensorflow as tf

def train_labels():
    y_train = {} 
    label_dir = os.listdir('./03211117_labelsR2')

    #d = {} # Keys = IDs for items
           # Values = binvox

    for label in label_dir:
        if(label.startswith('.')):
            continue
        binv = open('./03211117_labelsR2/'+label+'/model.binvox','rb')
        binvox_data = binvox_rw.read_as_3d_array(binv).data # binvox_data is 32x32x32
        y_train[label] = np.asarray(binvox_data)
        #tmp = tf.convert_to_tensor(binvox_data)
        #tmp = tf.cast(tmp,tf.float32)
        #y_train[label] = tmp

    return y_train
        


def train_data():
    x_train = {} #Keys = IDs for items, Values = 24 pictures
    data = os.listdir('./03211117R2')
    for item in data:
        tmp_im_array = []
        if(item.startswith('.')):
            continue
        for pic in os.listdir('./03211117R2/'+item+'/rendering'):
            if('.png' in pic): #If current item is a picture, store it
                im = np.array(Image.open('./03211117R2/'+item+'/rendering/'+pic))
                im = tf.random_crop(im,[127,127,3]) # Input images are [137,137,3]. Remove random 10x10 pixel area.
                tmp_im_array.append(im)
        x_train[item] = tmp_im_array
        tmp_im_array = []

    return x_train