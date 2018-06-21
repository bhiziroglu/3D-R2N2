import tensorflow as tf
import voxel
import numpy as np


def predict(w,image):
    #pred = tf.random_uniform([1, 32, 2, 32, 32])
    pred = np.random.rand(1,32,2,32,32)    
    voxel.voxel2obj('test_prediction.obj', pred[0, :, 1, :, :] > [0.4]) #0.4 Treshold


def train(w,x_train,y_train):
    print()
    ''' Do nothing '''