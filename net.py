from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import dataset
from tensorflow.python import debug as tf_debug
import voxel

# Training Parameters
learning_rate = 0.01
num_steps = 30000
batch_size = 128

display_step = 1000
examples_to_show = 10

# Network Parameters
num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 128 # 2nd layer num features (the latent dim)
num_input = 784 # MNIST data input (img shape: 28*28)
n_convfilter = [96, 128, 256, 256, 256, 256]
n_deconvfilter = [128, 128, 128, 64, 32, 2]
n_gru_vox = 4
n_fc_filters = [1024]
NUM_OF_IMAGES = 24

# tf Graph input (only pictures)
X = tf.placeholder(tf.float32, shape=[None, 127, 127, 3],name = "X")
Y = tf.placeholder(tf.float32, shape=[32,32,32,1,2],name = "Y")


initializer = tf.glorot_normal_initializer()

weights = {
    #Encoder Part
    'conv1a': tf.Variable(initializer([7,7,3,n_convfilter[0]])),
    'conv2a': tf.Variable(initializer([3,3,n_convfilter[0],n_convfilter[1]])),
    'conv3a': tf.Variable(initializer([3,3,n_convfilter[1],n_convfilter[2]])),
    'conv4a': tf.Variable(initializer([3,3,n_convfilter[2],n_convfilter[3]])),
    'conv5a': tf.Variable(initializer([3,3,n_convfilter[3],n_convfilter[4]])),
    'conv6a': tf.Variable(initializer([3,3,n_convfilter[4],n_convfilter[5]])),
    'fc7': tf.Variable(initializer([1,n_fc_filters[0]])),
    #Gru Translator
    'tmp_weight': tf.Variable(initializer([n_fc_filters[0],8192])),
    #Decoder Part
    'conv7a': tf.Variable(initializer([3,3,3,n_deconvfilter[0],n_deconvfilter[1]])),
    'conv8a': tf.Variable(initializer([3,3,3,n_deconvfilter[1],n_deconvfilter[2]])),
    'conv9a': tf.Variable(initializer([3,3,3,n_deconvfilter[2],n_deconvfilter[3]])),
    'conv10a': tf.Variable(initializer([3,3,3,n_deconvfilter[3],n_deconvfilter[4]])),
    'conv11a': tf.Variable(initializer([3,3,3,n_deconvfilter[4],n_deconvfilter[5]])),

}

def unpool(x): #unpool_3d_zero_filled
    # https://github.com/tensorflow/tensorflow/issues/2169
    out = tf.concat([x, tf.zeros_like(x)], 2)
    out = tf.concat([out, tf.zeros_like(out)], 1)
    out = tf.concat([out, tf.zeros_like(out)], 0)

    sh = x.get_shape().as_list()
    out_size = [sh[0]*2, sh[1] * 2, sh[2] * 2, -1, sh[4]]
    return tf.reshape(out, out_size)


biases = {
    #Encoder Part
    'conv1a': tf.Variable(tf.random_normal([1,1,1,n_convfilter[0]])),
    'conv2a': tf.Variable(tf.random_normal([1,1,1,n_convfilter[1]])),
    'conv3a': tf.Variable(tf.random_normal([1,1,1,n_convfilter[2]])),
    'conv4a': tf.Variable(tf.random_normal([1,1,1,n_convfilter[3]])),
    'conv5a': tf.Variable(tf.random_normal([1,1,1,n_convfilter[4]])),
    'conv6a': tf.Variable(tf.random_normal([1,1,1,n_convfilter[5]])),
    'fc7': tf.Variable(tf.random_normal([n_fc_filters[0]])),
    #Decoder Part
    'conv7a': tf.Variable(tf.random_normal([1,1,1,n_deconvfilter[1]])),
    'conv8a': tf.Variable(tf.random_normal([1,1,1,n_deconvfilter[2]])),
    'conv9a': tf.Variable(tf.random_normal([1,1,1,n_deconvfilter[3]])),
    'conv10a': tf.Variable(tf.random_normal([1,1,1,n_deconvfilter[4]])),
    'conv11a': tf.Variable(tf.random_normal([1,1,1,n_deconvfilter[5]]))
}

# Building the encoder
def encoder(x):

    with tf.name_scope("Encoder"):

        # Convolutional Layer #1
        conv1a = tf.nn.conv2d(input=X,filter=weights['conv1a'],strides=[1,1,1,1],padding="SAME")
        conv1a = tf.add(conv1a,biases['conv1a'])
        conv1a = tf.nn.max_pool(conv1a,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv1a = tf.nn.leaky_relu(conv1a,alpha=0.01)
        # [1, 64, 64, 96]

        # Convolutional Layer #2
        conv2a = tf.nn.conv2d(input=conv1a,filter=weights['conv2a'],strides=[1,1,1,1],padding="SAME")
        conv2a = tf.add(conv2a,biases['conv2a'])
        conv2a = tf.nn.max_pool(conv2a,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv2a = tf.nn.leaky_relu(conv2a,alpha=0.01)
        ''' !!!TODO!!!  (1, 32, 32, 128)   ->>>      Paper result size is (1, 33, 33, 128)'''

        # Convolutional Layer #3
        conv3a = tf.nn.conv2d(input=conv2a,filter=weights['conv3a'],strides=[1,1,1,1],padding="SAME")
        conv3a = tf.add(conv3a,biases['conv3a'])
        conv3a = tf.nn.max_pool(conv3a,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv3a = tf.nn.leaky_relu(conv3a,alpha=0.01)
        ''' !!!TODO!!!  (1, 16, 16, 256)   ->>>      Paper result size is (1, 17, 17, 256)'''

        # Convolutional Layer #4
        conv4a = tf.nn.conv2d(input=conv3a,filter=weights['conv4a'],strides=[1,1,1,1],padding="SAME")
        conv4a = tf.add(conv4a,biases['conv4a'])
        conv4a = tf.nn.max_pool(conv4a,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv4a = tf.nn.leaky_relu(conv4a,alpha=0.01)
        ''' !!!TODO!!!  (1, 8, 8, 256)   ->>>      Paper result size is (1, 9, 9, 256)'''

        # Convolutional Layer #5
        conv5a = tf.nn.conv2d(input=conv4a,filter=weights['conv5a'],strides=[1,1,1,1],padding="SAME")
        conv5a = tf.add(conv5a,biases['conv5a'])
        conv5a = tf.nn.max_pool(conv5a,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv5a = tf.nn.leaky_relu(conv5a,alpha=0.01)
        ''' !!!TODO!!!  (1, 4, 4, 256)   ->>>      Paper result size is (1, 5, 5, 256)'''

        # Convolutional Layer #6
        conv6a = tf.nn.conv2d(input=conv5a,filter=weights['conv6a'],strides=[1,1,1,1],padding="SAME")
        conv6a = tf.add(conv6a,biases['conv6a'])
        conv6a = tf.nn.max_pool(conv6a,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv6a = tf.nn.leaky_relu(conv6a,alpha=0.01)
        ''' !!!TODO!!!  (1, 2, 2, 256)   ->>>      Paper result size is (1, 3, 3, 256)'''

        # Flatten Layer
        #flat7 = tf.reshape(conv6a,[conv6a.shape[0],-1])
        flat7 = tf.layers.flatten(conv6a)
        ''' !!!TODO!!!  (1, 1024)   ->>>      Paper result size is (1, 2304)'''

        # FC Layer
        fc7 = tf.multiply(flat7,weights['fc7'])
        fc7 = tf.add(fc7,biases['fc7'])
        ''' w[15] was [1024] , now its [1,1024]. Which one is correct?'''
        # [N,1024]

    return fc7


# Building the decoder
def decoder(x):


    with tf.name_scope("Decoder"):

        x = tf.matmul(x,weights['tmp_weight'])

        #x = tf.reshape(x,[4,4,4,1,2])
        shape = tf.shape(x)
        ff = tf.reshape(x , [4,4,4,shape[0],128])

        unpool7 = unpool(ff)
        conv7a = tf.nn.conv3d(unpool7,weights['conv7a'],strides=[1,1,1,1,1],padding="SAME")
        conv7a = tf.add(conv7a,biases['conv7a'])
        conv7a = tf.nn.leaky_relu(conv7a,alpha=0.01)

        unpool8 = unpool(conv7a)
        conv8a = tf.nn.conv3d(unpool8,weights['conv8a'],strides=[1,1,1,1,1],padding="SAME")
        conv8a = tf.add(conv8a,biases['conv8a'])
        conv8a = tf.nn.leaky_relu(conv8a,alpha=0.01)

        unpool9 = unpool(conv8a)
        conv9a = tf.nn.conv3d(unpool9,weights['conv9a'],strides=[1,1,1,1,1],padding="SAME")
        conv9a = tf.add(conv9a,biases['conv9a'])
        conv9a = tf.nn.leaky_relu(conv9a,alpha=0.01)

        conv10a = tf.nn.conv3d(conv9a,weights['conv10a'],strides=[1,1,1,1,1],padding="SAME")
        conv10a = tf.add(conv10a,biases['conv10a'])
        conv10a = tf.nn.leaky_relu(conv10a,alpha=0.01)

        conv11a = tf.nn.conv3d(conv10a,weights['conv11a'],strides=[1,1,1,1,1],padding="SAME")
        conv11a = tf.add(conv11a,biases['conv11a'])
        ''' (32, 32, 32, 1, 2) '''

    return conv11a

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = Y

# Define loss and optimizer, minimize the squared error
#loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
loss = tf.reduce_mean(
            tf.reduce_sum(-y_true * y_pred ,axis=4,keepdims=True)+
            tf.log(tf.reduce_sum(tf.exp(y_pred),axis=4,keepdims=True))
        )

optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    #sess = tf_debug.TensorBoardDebugWrapperSession(sess, "Berkan-MacBook-Pro.local:4334")

    # Run the initializer
    sess.run(init)

    train_writer = tf.summary.FileWriter('./train',
                    sess.graph)

    x_train = dataset.train_data()
    y_train = dataset.train_labels()

    i = 0



    while(i<num_steps):

        for image_hash in x_train.keys():

            i+=1
            ims = []
            for image in x_train[image_hash]:
                ims.append(image)
                break

            ims = tf.convert_to_tensor(ims)
            ims = tf.reshape(ims,[-1,127,127,3])
            ims = ims.eval()


            vox = tf.convert_to_tensor(y_train[image_hash])
            vox = tf.cast(vox,tf.float32)
            vox = vox.eval()

            batch_voxel = np.zeros((32,32,32,1,2), dtype=float)
            batch_voxel[:,:,:,0,0]= vox < 1
            batch_voxel[:,:,:,0,1]= vox

            # Run optimization op (backprop) and cost op (to get loss value)
            _, l = sess.run([optimizer, loss], feed_dict={X: ims, Y: batch_voxel})

            if(i%2==0):
                print("Creating prediction objects.")
                pred = sess.run([y_pred], feed_dict={X: ims, Y: batch_voxel})
                pred = np.array(pred)
                pred = tf.convert_to_tensor(pred[0,:,:,:,:,:])

                exp_x = tf.exp(pred)
                sum_exp_x = tf.reduce_sum(exp_x,axis=4,keepdims=True)

                pred = exp_x / sum_exp_x
                pred = pred.eval()

                pred_name = "test_pred_"+str(i)+".obj"
                voxel.voxel2obj(pred_name,pred[:,:,:,0,0])

            # Display logs per step
            print('Step %i: Loss: %f' % (i, l))