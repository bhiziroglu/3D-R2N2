import numpy as np
import argparse
import pprint
import logging
import multiprocessing as mp

import tensorflow as tf
import net
import dataset

def main():
    
    parser = argparse.ArgumentParser(description='Make everything 3D')
    parser.add_argument('--batch-size',dest='batch_size',help='Batch size',default=120,type=int)
    parser.add_argument('--iter',dest='iter',help='Number of iterations',default=1000,type=int)
    parser.add_argument('--weights', dest='weights', help='Pre-trained weights', default=None)
    args = parser.parse_args()
    print('Called with args:' , args)

    w = []
    x_train = dataset.train_data()
    y_train = dataset.train_labels()

    #print(len(x_train['1a92ca1592aefb2f9531a714ad5bf7d5']))

    net.train(w,x_train,y_train) # Train the network

    # Test network after training is finished,
    image = 'fake_image'

    net.predict(w,image)


    del x_train
    del y_train
    del args






if __name__ == '__main__':
    main()