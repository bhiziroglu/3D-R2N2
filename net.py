import tensorflow as tf
import voxel
import numpy as np



def predict(w,image):
    #pred = tf.random_uniform([1, 32, 2, 32, 32])
    
    # Paper - Implementation
    #pred = np.random.rand(1,32,2,32,32)    
    #voxel.voxel2obj('test_prediction.obj', pred[0, :, 1, :, :] > [0.4]) #0.4 Treshold
    
    pred = np.random.rand(32,32,32)
    voxel.voxel2obj('test_pred.obj',pred[:, :, :] > [0.4])
    return pred


def train(w,x_train,y_train):
    print()
    '''Do nothing'''
    for images in x_train.keys():
        ims = [] # Concatenate N images
        for image in x_train[images]:
            #image = tf.convert_to_tensor(image) # (127,127,3)
            #image = tf.reshape(image,[3,127,127])
            ims.append(image)   
        tmp = encoder(w,ims)
        print(tmp.shape)



def loss(w,x,y):
    #w []
    #x [127,127,3]
    #y [32,32,32]
    pred = predict(w,x) # [32,32,32]
    return 1 # [1,32,2,32,32] -> Only take voxel values



def encoder(w,ims):
    # (multi_views, self.batch_size, 3, self.img_h, self.img_w),
    img_w = 127
    img_h = 127
    n_gru_vox = 4
    # n_vox = self.n_vox

    n_convfilter = [96, 128, 256, 256, 256, 256]
    n_fc_filters = [1024]
    n_deconvfilter = [128, 128, 128, 64, 32, 2]
    input_shape = (1, 3, img_w, img_h) # (batchsize, rgb, width , height)


    # Input Layer
    ims = tf.convert_to_tensor(ims) # 24 Images are stored in a list, convert them to Tensor
    ims = ims[0:1,:,:,:] # Take out 1 image
    #ims = tf.transpose(ims,[0,3,1,2]) # (1, rgb, width, height) -> Take RGB to beginning
    input_layer = tf.cast(ims, tf.float32)

    # Convolutional Layer #1
    conv1a = tf.nn.conv2d(input=input_layer,filter=w[0],strides=[1,1,1,1],padding="SAME")
    conv1b = tf.nn.conv2d(input=conv1a,filter=w[1],strides=[1,1,1,1],padding="SAME")
    pool1 = tf.nn.max_pool(conv1b,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    # [1, 64, 64, 96]

    # Convolutional Layer #2
    conv2a = tf.nn.conv2d(input=pool1,filter=w[2],strides=[1,1,1,1],padding="SAME")
    conv2b = tf.nn.conv2d(input=conv2a,filter=w[3],strides=[1,1,1,1],padding="SAME")
    conv2c = tf.nn.conv2d(input=pool1,filter=w[4],strides=[1,1,1,1],padding="SAME")
    res2 = tf.add(conv2b,conv2c)
    pool2 = tf.nn.max_pool(res2,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    ''' !!!TODO!!!  (1, 32, 32, 128)   ->>>      Paper result size is (1, 33, 33, 128)'''

    # Convolutional Layer #3
    conv3a = tf.nn.conv2d(input=pool2,filter=w[5],strides=[1,1,1,1],padding="SAME")
    conv3b = tf.nn.conv2d(input=conv3a,filter=w[6],strides=[1,1,1,1],padding="SAME")
    conv3c = tf.nn.conv2d(input=pool2,filter=w[7],strides=[1,1,1,1],padding="SAME")
    res3 = tf.add(conv3b,conv3c)
    pool3 = tf.nn.max_pool(res3,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    ''' !!!TODO!!!  (1, 16, 16, 256)   ->>>      Paper result size is (1, 17, 17, 256)'''

    # Convolutional Layer #4
    conv4a = tf.nn.conv2d(input=pool3,filter=w[8],strides=[1,1,1,1],padding="SAME")
    conv4b = tf.nn.conv2d(input=conv4a,filter=w[9],strides=[1,1,1,1],padding="SAME")
    pool4 = tf.nn.max_pool(conv4b,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    ''' !!!TODO!!!  (1, 8, 8, 256)   ->>>      Paper result size is (1, 9, 9, 256)'''
   
    # Convolutional Layer #5
    conv5a = tf.nn.conv2d(input=pool4,filter=w[10],strides=[1,1,1,1],padding="SAME")
    conv5b = tf.nn.conv2d(input=conv5a,filter=w[11],strides=[1,1,1,1],padding="SAME")
    conv5c = tf.nn.conv2d(input=pool4,filter=w[12],strides=[1,1,1,1],padding="SAME")
    res5 = tf.add(conv5b,conv5c)
    pool5 = tf.nn.max_pool(res5,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    ''' !!!TODO!!!  (1, 4, 4, 256)   ->>>      Paper result size is (1, 5, 5, 256)'''
   
   # Convolutional Layer #6
    conv6a = tf.nn.conv2d(input=pool5,filter=w[13],strides=[1,1,1,1],padding="SAME")
    conv6b = tf.nn.conv2d(input=conv6a,filter=w[14],strides=[1,1,1,1],padding="SAME")
    pool6 = tf.nn.max_pool(conv6b,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    ''' !!!TODO!!!  (1, 2, 2, 256)   ->>>      Paper result size is (1, 3, 3, 256)'''
   
    # Flatten Layer
    flat7 = tf.reshape(pool6,[pool6.shape[0],-1])
    ''' !!!TODO!!!  (1, 1024)   ->>>      Paper result size is (1, 2304)'''

    # FC Layer
    fc7 = tf.multiply(flat7,w[15])
    # [1,1024]

    return fc7

