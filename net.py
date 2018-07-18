from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import dataset
from tensorflow.python import debug as tf_debug
import voxel
from random import shuffle

# Training Parameters
learning_rate = 0.01
num_steps = 1000
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
p_H = tf.placeholder(tf.float32, [n_gru_vox, n_gru_vox, n_gru_vox, 1, n_deconvfilter[0]], name="p_H")
Y = tf.placeholder(tf.float32, shape=[32,32,32,1,2],name = "Y")
G = tf.placeholder(tf.float32, shape=[4,4,4,1,128],name = "GRU_OUT")
#D = tf.placeholder(tf.float32, shape=[32,32,32,1,2],name = "DECODER_OUT")

initializer = tf.glorot_normal_initializer(seed=123)

weights = {
    #Encoder Part
    'conv1a': tf.Variable(initializer([7,7,3,n_convfilter[0]])),
    'conv2a': tf.Variable(initializer([3,3,n_convfilter[0],n_convfilter[1]])),
    'conv3a': tf.Variable(initializer([3,3,n_convfilter[1],n_convfilter[2]])),
    'conv4a': tf.Variable(initializer([3,3,n_convfilter[2],n_convfilter[3]])),
    'conv5a': tf.Variable(initializer([3,3,n_convfilter[3],n_convfilter[4]])),
    'conv6a': tf.Variable(initializer([3,3,n_convfilter[4],n_convfilter[5]])),
    'fc7': tf.Variable(initializer([1,n_fc_filters[0]])),
    #Gru Part
    'w_update': tf.Variable(initializer([1024,8192])), #
    'update_gate': tf.Variable(initializer([3,3,3,n_deconvfilter[0],n_deconvfilter[0]])),
    'reset_gate': tf.Variable(initializer([3, 3, 3, n_deconvfilter[0], n_deconvfilter[0]])),
    'tanh_reset': tf.Variable(initializer([3, 3, 3, n_deconvfilter[0], n_deconvfilter[0]])),
    #'prev_s': tf.Variable(tf.zeros([n_gru_vox, n_gru_vox, n_gru_vox, 1, n_deconvfilter[0]])),
    #Decoder Part
    'conv7a': tf.Variable(initializer([3,3,3,n_deconvfilter[0],n_deconvfilter[1]])),
    'conv8a': tf.Variable(initializer([3,3,3,n_deconvfilter[1],n_deconvfilter[2]])),
    'conv9a': tf.Variable(initializer([3,3,3,n_deconvfilter[2],n_deconvfilter[3]])),
    'conv10a': tf.Variable(initializer([3,3,3,n_deconvfilter[3],n_deconvfilter[4]])),
    'conv11a': tf.Variable(initializer([3,3,3,n_deconvfilter[4],n_deconvfilter[5]]))
}

biases = {
    #Encoder Part
    'conv1a':       tf.Variable(tf.random_normal([1,1,1,n_convfilter[0]])),
    'conv2a':       tf.Variable(tf.random_normal([1,1,1,n_convfilter[1]])),
    'conv3a':       tf.Variable(tf.random_normal([1,1,1,n_convfilter[2]])),
    'conv4a':       tf.Variable(tf.random_normal([1,1,1,n_convfilter[3]])),
    'conv5a':       tf.Variable(tf.random_normal([1,1,1,n_convfilter[4]])),
    'conv6a':       tf.Variable(tf.random_normal([1,1,1,n_convfilter[5]])),
    'fc7':          tf.Variable(tf.random_normal([n_fc_filters[0]])),
    #Gru Part
    'w_update':     tf.Variable(tf.random_normal([8192])),
    'update_gate':  tf.Variable(tf.random_normal([1,1,1,n_deconvfilter[0]])),
    'reset_gate':   tf.Variable(tf.random_normal([1,1,1,n_deconvfilter[0]])),
    'tanh_reset':   tf.Variable(tf.random_normal([1,1,1,n_deconvfilter[0]])),
    #Decoder Part
    'conv7a':       tf.Variable(tf.random_normal([1,1,1,n_deconvfilter[1]])),
    'conv8a':       tf.Variable(tf.random_normal([1,1,1,n_deconvfilter[2]])),
    'conv9a':       tf.Variable(tf.random_normal([1,1,1,n_deconvfilter[3]])),
    'conv10a':      tf.Variable(tf.random_normal([1,1,1,n_deconvfilter[4]])),
    'conv11a':      tf.Variable(tf.random_normal([1,1,1,n_deconvfilter[5]]))
}


def unpool(x): #unpool_3d_zero_filled
    # https://github.com/tensorflow/tensorflow/issues/2169
    out = tf.concat([x, tf.zeros_like(x)], 2)
    out = tf.concat([out, tf.zeros_like(out)], 1)
    out = tf.concat([out, tf.zeros_like(out)], 0)

    sh = x.get_shape().as_list()
    out_size = [sh[0]*2, sh[1] * 2, sh[2] * 2, -1, sh[4]]
    return tf.reshape(out, out_size)


def gru():

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


    with tf.name_scope("GRU"):

        prev_hidden = p_H

        fc_layer = tf.matmul(fc7, weights['w_update'])  # [1,1024]x[1024,8192] // FC LAYER FOR UPDATE GATE
        fc_layer = tf.reshape(fc_layer, [4, 4, 4, -1, 128])  # [1,4,128,4,4]

        t_x_s_update = tf.nn.conv3d(prev_hidden, weights['update_gate'], strides=[1, 1, 1, 1, 1], padding="SAME") + fc_layer
        t_x_s_update = tf.add(t_x_s_update, biases['update_gate']) #Bias
        t_x_s_reset = tf.nn.conv3d(prev_hidden, weights['reset_gate'], strides=[1, 1, 1, 1, 1], padding="SAME") + fc_layer
        t_x_s_reset = tf.add(t_x_s_reset, biases['reset_gate']) #Bias

        update_gate = tf.sigmoid(t_x_s_update)

        complement_update_gate = 1 - update_gate
        reset_gate = tf.sigmoid(t_x_s_reset)

        rs = reset_gate * prev_hidden
        t_x_rs = tf.nn.conv3d(rs, weights['tanh_reset'], strides=[1, 1, 1, 1, 1], padding="SAME") + fc_layer
        t_x_rs = tf.add(t_x_rs, biases['tanh_reset']) #Bias
        tanh_t_x_rs = tf.tanh(t_x_rs)

        gru_out = update_gate * prev_hidden + complement_update_gate * tanh_t_x_rs


    return gru_out

# Building the decoder
def decoder():


    with tf.name_scope("Decoder"):

        #x = tf.reshape(x,[4,4,4,1,2])
        unpool7 = unpool(G)
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

        exp_x = tf.exp(conv11a)  # 32, 32, 32, 1 ,2
        sum_exp_x = tf.reduce_sum(exp_x, axis=4, keepdims=True)  # 32, 32, 32, 1, 1

        loss = tf.reduce_mean(
            tf.reduce_sum(-Y * conv11a, axis=4, keepdims=True) +
            tf.log(sum_exp_x)
        )


    return conv11a, loss

# Construct model
gru_op = gru()
output, loss = decoder()

optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
#optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    #sess = tf_debug.TensorBoardDebugWrapperSession(sess, "Berkan-MacBook-Pro.local:4334")

    # Run the initializer
    sess.run(init)

    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter('./train',sess.graph)

    x_train = dataset.train_data()
    y_train = dataset.train_labels()

    i = 0

    prev_state = np.zeros([n_gru_vox, n_gru_vox, n_gru_vox, 1, n_deconvfilter[0]])

    while(i<num_steps):


        for image_hash in x_train.keys():

            i+=1

            images = x_train[image_hash]
            shuffle(images) #Shuffle views
            for image in images:
                #ims = tf.convert_to_tensor(image)
                #ims = tf.reshape(ims,[-1,127,127,3])
                #ims = ims.eval()
                ims = tf.ones([1,127,127,3])
                ims = ims.eval()

                prev_s = sess.run([gru_op], feed_dict={X: ims, p_H: prev_state})
                prev_s = np.array(prev_s)
                prev_s = prev_s[0,:,:,:,:,:]
                prev_state = prev_s


            vox = tf.convert_to_tensor(y_train[image_hash])
            vox = tf.cast(vox,tf.float64)
            print("XDXD KLKL XDXD")
            print(sess.run(tf.reduce_mean(vox)))
            vox = vox.eval()

            batch_voxel = np.ones((32,32,32,1,2), dtype=int)

            batch_voxel[:,:,:,0,1] = vox
            batch_voxel[:,:,:,0,0] = (tf.ones_like(vox) - vox).eval()

            #decoder_out, loss = sess.run([decoder_op], feed_dict={G: prev_state, Y: batch_voxel})

            # Run optimization op (backprop) and cost op (to get loss value)
            l, o, _ = sess.run([loss, output, optimizer], feed_dict={G: prev_state, Y: batch_voxel})

            #train_writer.add_summary(summary, i)

            if(i%1==0):

                print("Creating prediction objects.")

                exp_x = tf.exp(o) #32, 32, 32, 1 ,2
                sum_exp_x = tf.reduce_sum(exp_x,axis=4,keepdims=True) # 32, 32, 32, 1, 1

                pred = exp_x / sum_exp_x

                pred = pred.eval()


                pred_name = "test_pred_"+str(i)+".obj"
                pred_name2 = "test_pred_"+str(i)+"_XD.obj"


                voxel.voxel2obj(pred_name2,pred[:,:,:,0,1])
                voxel.voxel2obj(pred_name,pred[:,:,:,0,0])
            



            # Display logs per step
            print('Step %i: Loss: %f' % (i, l))
