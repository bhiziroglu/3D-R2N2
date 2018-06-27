import numpy as np
import argparse
import pprint
import logging
import multiprocessing as mp

import tensorflow as tf
import net
import dataset

tf.logging.set_verbosity(tf.logging.INFO)

def main():
    
    parser = argparse.ArgumentParser(description='Make everything 3D')
    parser.add_argument('--batch-size',dest='batch_size',help='Batch size',default=120,type=int)
    parser.add_argument('--iter',dest='iter',help='Number of iterations',default=1000,type=int)
    parser.add_argument('--weights', dest='weights', help='Pre-trained weights', default=None)
    args = parser.parse_args()
    print('Called with args:' , args)

    n_convfilter = [96, 128, 256, 256, 256, 256]
    n_fc_filters = [1024]
    n_deconvfilter = [128, 128, 128, 64, 32, 2]

    w0 = tf.Variable(tf.truncated_normal([7,7,3,n_convfilter[0]], stddev=0.5),name="w0")
    w1 = tf.Variable(tf.truncated_normal([3,3,n_convfilter[0],n_convfilter[0]], stddev=0.5), name="w1")
    w2 = tf.Variable(tf.truncated_normal([3,3,n_convfilter[0],n_convfilter[1]], stddev=0.5), name="w2")
    w3 = tf.Variable(tf.truncated_normal([3,3,n_convfilter[1],n_convfilter[1]], stddev=0.5), name="w3")
    w4 = tf.Variable(tf.truncated_normal([1,1,n_convfilter[0],n_convfilter[1]], stddev=0.5), name="w4")
    w5 = tf.Variable(tf.truncated_normal([3,3,n_convfilter[1],n_convfilter[2]], stddev=0.5), name="w5")
    w6 = tf.Variable(tf.truncated_normal([3,3,n_convfilter[2],n_convfilter[2]], stddev=0.5), name="w6")
    w7 = tf.Variable(tf.truncated_normal([1,1,n_convfilter[1],n_convfilter[2]], stddev=0.5), name="w7")
    w8 = tf.Variable(tf.truncated_normal([3,3,n_convfilter[2],n_convfilter[3]], stddev=0.5), name="w8")
    w9 = tf.Variable(tf.truncated_normal([3,3,n_convfilter[3],n_convfilter[3]], stddev=0.5), name="w9")
    w10 = tf.Variable(tf.truncated_normal([3,3,n_convfilter[3],n_convfilter[4]], stddev=0.5), name="w10")
    w11 = tf.Variable(tf.truncated_normal([3,3,n_convfilter[4],n_convfilter[4]], stddev=0.5), name="w11")
    w12 = tf.Variable(tf.truncated_normal([1,1,n_convfilter[4],n_convfilter[4]], stddev=0.5), name="w12")
    w13 = tf.Variable(tf.truncated_normal([3,3,n_convfilter[4],n_convfilter[5]], stddev=0.5), name="w13")
    w14 = tf.Variable(tf.truncated_normal([3,3,n_convfilter[5],n_convfilter[5]], stddev=0.5), name="w14")
    w15 = tf.Variable(tf.truncated_normal([n_fc_filters[0]]), name="w15")
    w = [w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15]
    x_train = dataset.train_data()
    y_train = dataset.train_labels()

    net.train(w,x_train,y_train) # Train the network



    del x_train
    del y_train
    del args
    del w1






if __name__ == '__main__':
    main()