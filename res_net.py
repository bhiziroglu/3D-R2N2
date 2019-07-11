from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import dataset
from tensorflow.python import debug as tf_debug
import voxel
from random import shuffle

# Training Parameters
num_steps = 1000

# Network Parameters
n_convfilter = [96, 128, 256, 256, 256, 256]
n_deconvfilter = [128, 128, 128, 64, 32, 2]
n_gru_vox = 4
n_fc_filters = [1024]
NUM_OF_IMAGES = 24

# tf Graph input (only pictures)
X = tf.placeholder(tf.float32, shape=[None, 127, 127, 3], name="X")
p_H = tf.placeholder(tf.float32, [n_gru_vox, n_gru_vox, n_gru_vox, 1, n_deconvfilter[0]], name="p_H")
Y = tf.placeholder(tf.float32, shape=[32, 32, 32, 1, 2], name="Y")
G = tf.placeholder(tf.float32, shape=[4, 4, 4, 1, 128], name="GRU_OUT")

initializer = tf.glorot_normal_initializer(seed=123)

weights = {
    # Encoder Part
    'conv1a': tf.Variable(initializer([7, 7, 3, n_convfilter[0]])),
    'conv1b': tf.Variable(initializer([3, 3, n_convfilter[0], n_convfilter[0]])),

    'conv2a': tf.Variable(initializer([3, 3, n_convfilter[0], n_convfilter[1]])),
    'conv2b': tf.Variable(initializer([3, 3, n_convfilter[1], n_convfilter[1]])),
    'conv2c': tf.Variable(initializer([1, 1, n_convfilter[0], n_convfilter[1]])),

    'conv3a': tf.Variable(initializer([3, 3, n_convfilter[1], n_convfilter[2]])),
    'conv3b': tf.Variable(initializer([3, 3, n_convfilter[2], n_convfilter[2]])),
    'conv3c': tf.Variable(initializer([1, 1, n_convfilter[1], n_convfilter[2]])),

    'conv4a': tf.Variable(initializer([3, 3, n_convfilter[2], n_convfilter[3]])),
    'conv4b': tf.Variable(initializer([3, 3, n_convfilter[3], n_convfilter[3]])),
    'conv4c': tf.Variable(initializer([3, 3, n_convfilter[3], n_convfilter[4]])),

    'conv5a': tf.Variable(initializer([3, 3, n_convfilter[3], n_convfilter[4]])),
    'conv5b': tf.Variable(initializer([3, 3, n_convfilter[4], n_convfilter[4]])),
    'conv5c': tf.Variable(initializer([3, 3, n_convfilter[4], n_convfilter[4]])),

    'conv6a': tf.Variable(initializer([3, 3, n_convfilter[4], n_convfilter[5]])),
    'conv6b': tf.Variable(initializer([3, 3, n_convfilter[5], n_convfilter[5]])),

    'fc7': tf.Variable(initializer([1, n_fc_filters[0]])),

    # Gru Part
    'w_update': tf.Variable(initializer([1024, 8192])),  #
    'update_gate': tf.Variable(initializer([3, 3, 3, n_deconvfilter[0], n_deconvfilter[0]])),
    'reset_gate': tf.Variable(initializer([3, 3, 3, n_deconvfilter[0], n_deconvfilter[0]])),
    'tanh_reset': tf.Variable(initializer([3, 3, 3, n_deconvfilter[0], n_deconvfilter[0]])),
    
    # Decoder Part
    'conv7a': tf.Variable(initializer([3, 3, 3, n_deconvfilter[0], n_deconvfilter[1]])),
    'conv7b': tf.Variable(initializer([3, 3, 3, n_deconvfilter[1], n_deconvfilter[1]])),

    'conv8a': tf.Variable(initializer([3, 3, 3, n_deconvfilter[1], n_deconvfilter[2]])),
    'conv8b': tf.Variable(initializer([3, 3, 3, n_deconvfilter[2], n_deconvfilter[2]])),

    'conv9a': tf.Variable(initializer([3, 3, 3, n_deconvfilter[2], n_deconvfilter[3]])),
    'conv9b': tf.Variable(initializer([3, 3, 3, n_deconvfilter[3], n_deconvfilter[3]])),
    'conv9c': tf.Variable(initializer([1, 1, 1, n_deconvfilter[2], n_deconvfilter[3]])),

    'conv10a': tf.Variable(initializer([3, 3, 3, n_deconvfilter[3], n_deconvfilter[4]])),
    'conv10b': tf.Variable(initializer([3, 3, 3, n_deconvfilter[4], n_deconvfilter[4]])),
    'conv10c': tf.Variable(initializer([3, 3, 3, n_deconvfilter[4], n_deconvfilter[4]])),

    'conv11a': tf.Variable(initializer([3, 3, 3, n_deconvfilter[4], n_deconvfilter[5]]))
}

biases = {
    # Encoder Part
    'conv1a': tf.Variable(tf.random_normal([1, 1, 1, n_convfilter[0]])),
    'conv1b': tf.Variable(tf.random_normal([1, 1, 1, n_convfilter[1]])),

    'conv2a': tf.Variable(tf.random_normal([1, 1, 1, n_convfilter[1]])),
    'conv2b': tf.Variable(tf.random_normal([1, 1, 1, n_convfilter[1]])),
    'conv2c': tf.Variable(tf.random_normal([1, 1, 1, n_convfilter[1]])),

    'conv3a': tf.Variable(tf.random_normal([1, 1, 1, n_convfilter[2]])),
    'conv3b': tf.Variable(tf.random_normal([1, 1, 1, n_convfilter[2]])),
    'conv3c': tf.Variable(tf.random_normal([1, 1, 1, n_convfilter[2]])),
    
    'conv4a': tf.Variable(tf.random_normal([1, 1, 1, n_convfilter[3]])),
    'conv4b': tf.Variable(tf.random_normal([1, 1, 1, n_convfilter[3]])),
    'conv4c': tf.Variable(tf.random_normal([1, 1, 1, n_convfilter[3]])),


    'conv5a': tf.Variable(tf.random_normal([1, 1, 1, n_convfilter[4]])),
    'conv5b': tf.Variable(tf.random_normal([1, 1, 1, n_convfilter[4]])),
    'conv5c': tf.Variable(tf.random_normal([1, 1, 1, n_convfilter[4]])),


    'conv6a': tf.Variable(tf.random_normal([1, 1, 1, n_convfilter[5]])),
    'conv6b': tf.Variable(tf.random_normal([1, 1, 1, n_convfilter[5]])),


    'fc7': tf.Variable(tf.random_normal([n_fc_filters[0]])),

    # Gru Part
    'w_update': tf.Variable(tf.random_normal([8192])),
    'update_gate': tf.Variable(tf.random_normal([1, 1, 1, n_deconvfilter[0]])),
    'reset_gate': tf.Variable(tf.random_normal([1, 1, 1, n_deconvfilter[0]])),
    'tanh_reset': tf.Variable(tf.random_normal([1, 1, 1, n_deconvfilter[0]])),


    # Decoder Part
    'conv7a': tf.Variable(tf.random_normal([1, 1, 1, n_deconvfilter[1]])),
    'conv7b': tf.Variable(tf.random_normal([1, 1, 1, n_deconvfilter[1]])),

    'conv8a': tf.Variable(tf.random_normal([1, 1, 1, n_deconvfilter[2]])),
    'conv8b': tf.Variable(tf.random_normal([1, 1, 1, n_deconvfilter[2]])),

    'conv9a': tf.Variable(tf.random_normal([1, 1, 1, n_deconvfilter[3]])),
    'conv9b': tf.Variable(tf.random_normal([1, 1, 1, n_deconvfilter[3]])),
    'conv9c': tf.Variable(tf.random_normal([1, 1, 1, n_deconvfilter[3]])),


    'conv10a': tf.Variable(tf.random_normal([1, 1, 1, n_deconvfilter[4]])),
    'conv10b': tf.Variable(tf.random_normal([1, 1, 1, n_deconvfilter[4]])),
    'conv10c': tf.Variable(tf.random_normal([1, 1, 1, n_deconvfilter[4]])),


    'conv11a': tf.Variable(tf.random_normal([1, 1, 1, n_deconvfilter[5]]))
}


def unpool(x):  # unpool_3d_zero_filled
    # https://github.com/tensorflow/tensorflow/issues/2169
    out = tf.concat([x, tf.zeros_like(x)], 2)
    out = tf.concat([out, tf.zeros_like(out)], 1)
    out = tf.concat([out, tf.zeros_like(out)], 0)

    sh = x.get_shape().as_list()
    out_size = [sh[0] * 2, sh[1] * 2, sh[2] * 2, -1, sh[4]]
    return tf.reshape(out, out_size)


def gru():
    with tf.name_scope("Encoder"):

        conv1a = tf.nn.conv2d(input=X, filter=weights['conv1a'], strides=[1, 1, 1, 1], padding="SAME")
        conv1a = tf.add(conv1a, biases['conv1a'])
        conv1a = tf.nn.leaky_relu(conv1a, alpha=0.01)

        conv1b = tf.nn.conv2d(input=conv1a, filter=weights['conv1b'], strides=[1, 1, 1, 1], padding="SAME")
        conv1b = tf.nn.max_pool(conv1b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv1b = tf.nn.leaky_relu(conv1b, alpha=0.01)

        # Convolutional Layer #2
        conv2a = tf.nn.conv2d(input=conv1b, filter=weights['conv2a'], strides=[1, 1, 1, 1], padding="SAME")
        conv2a = tf.add(conv2a, biases['conv2a'])
        conv2a = tf.nn.leaky_relu(conv2a, alpha=0.01)


        conv2b = tf.nn.conv2d(input=conv2a, filter=weights['conv2b'], strides=[1, 1, 1, 1], padding="SAME")
        conv2b = tf.add(conv2b, biases['conv2b'])
        conv2b = tf.nn.leaky_relu(conv2b, alpha=0.01)


        conv2c = tf.nn.conv2d(input=conv1b, filter=weights['conv2c'], strides=[1, 1, 1, 1], padding="SAME")
        conv2c = tf.add(conv2c, biases['conv2c'])
        conv2c = tf.nn.leaky_relu(conv2c, alpha=0.01)

        conv2c = tf.add(conv2c,conv2b)

        conv2c = tf.nn.max_pool(conv2c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


        # Convolutional Layer #3
        conv3a = tf.nn.conv2d(input=conv2c, filter=weights['conv3a'], strides=[1, 1, 1, 1], padding="SAME")
        conv3a = tf.add(conv3a, biases['conv3a'])
        conv3a = tf.nn.leaky_relu(conv3a, alpha=0.01)

        conv3b = tf.nn.conv2d(input=conv3a, filter=weights['conv3b'], strides=[1, 1, 1, 1], padding="SAME")
        conv3b = tf.add(conv3b, biases['conv3b'])
        conv3b = tf.nn.leaky_relu(conv3b, alpha=0.01)

        conv3c = tf.nn.conv2d(input=conv2c, filter=weights['conv3c'], strides=[1, 1, 1, 1], padding="SAME")
        conv3c = tf.add(conv3c, biases['conv3c'])
        conv3c = tf.nn.leaky_relu(conv3c, alpha=0.01)

        conv3c = tf.add(conv3c,conv3b)

        conv3c = tf.nn.max_pool(conv3c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # Convolutional Layer #4
        conv4a = tf.nn.conv2d(input=conv3c, filter=weights['conv4a'], strides=[1, 1, 1, 1], padding="SAME")
        conv4a = tf.add(conv4a, biases['conv4a'])
        conv4a = tf.nn.leaky_relu(conv4a, alpha=0.01)


        conv4b = tf.nn.conv2d(input=conv4a, filter=weights['conv4b'], strides=[1, 1, 1, 1], padding="SAME")
        conv4b = tf.add(conv4b, biases['conv4b'])
        conv4b = tf.nn.leaky_relu(conv4a, alpha=0.01)

        conv4b = tf.nn.max_pool(conv4b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


        # Convolutional Layer #5
        conv5a = tf.nn.conv2d(input=conv4b, filter=weights['conv5a'], strides=[1, 1, 1, 1], padding="SAME")
        conv5a = tf.add(conv5a, biases['conv5a'])
        conv5a = tf.nn.leaky_relu(conv5a, alpha=0.01)

        conv5b = tf.nn.conv2d(input=conv5a, filter=weights['conv5b'], strides=[1, 1, 1, 1], padding="SAME")
        conv5b = tf.add(conv5b, biases['conv5b'])       
        conv5b = tf.nn.leaky_relu(conv5b, alpha=0.01)

        
        conv5c = tf.nn.conv2d(input=conv5b, filter=weights['conv5c'], strides=[1, 1, 1, 1], padding="SAME")
        conv5c = tf.add(conv5c, biases['conv5c'])       
        conv5c = tf.nn.leaky_relu(conv5c, alpha=0.01)

        conv5c = tf.add(conv5c,conv5b)

        conv5c = tf.nn.max_pool(conv5c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # Convolutional Layer #6
        conv6a = tf.nn.conv2d(input=conv5c, filter=weights['conv6a'], strides=[1, 1, 1, 1], padding="SAME")
        conv6a = tf.add(conv6a, biases['conv6a'])
        conv6a = tf.nn.leaky_relu(conv6a, alpha=0.01)
        
        conv6b = tf.nn.conv2d(input=conv6a, filter=weights['conv6b'], strides=[1, 1, 1, 1], padding="SAME")
        conv6b = tf.add(conv6b, biases['conv6b'])
        conv6b = tf.nn.leaky_relu(conv5c, alpha=0.01)

        conv6b = tf.nn.max_pool(conv6b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
     
        # Flatten Layer
        flat7 = tf.layers.flatten(conv6b)

        # FC Layer
        fc7 = tf.multiply(flat7, weights['fc7'])
        fc7 = tf.add(fc7, biases['fc7'])


    with tf.name_scope("GRU"):
        prev_hidden = p_H

        fc_layer = tf.matmul(fc7, weights['w_update']) 
        fc_layer = tf.reshape(fc_layer, [4, 4, 4, -1, 128]) 

        t_x_s_update = tf.nn.conv3d(prev_hidden, weights['update_gate'], strides=[1, 1, 1, 1, 1],
                                    padding="SAME") + fc_layer
        t_x_s_update = tf.add(t_x_s_update, biases['update_gate']) 
        t_x_s_reset = tf.nn.conv3d(prev_hidden, weights['reset_gate'], strides=[1, 1, 1, 1, 1],
                                   padding="SAME") + fc_layer
        t_x_s_reset = tf.add(t_x_s_reset, biases['reset_gate']) 

        update_gate = tf.sigmoid(t_x_s_update)

        complement_update_gate = 1 - update_gate
        reset_gate = tf.sigmoid(t_x_s_reset)

        rs = reset_gate * prev_hidden
        t_x_rs = tf.nn.conv3d(rs, weights['tanh_reset'], strides=[1, 1, 1, 1, 1], padding="SAME") + fc_layer
        t_x_rs = tf.add(t_x_rs, biases['tanh_reset'])  
        tanh_t_x_rs = tf.tanh(t_x_rs)

        gru_out = update_gate * prev_hidden + complement_update_gate * tanh_t_x_rs

    return gru_out


# Building the decoder
def decoder():
    with tf.name_scope("Decoder"):

        unpool7 = unpool(G)

        conv7a = tf.nn.conv3d(unpool7, weights['conv7a'], strides=[1, 1, 1, 1, 1], padding="SAME")
        conv7a = tf.add(conv7a, biases['conv7a'])
        conv7a = tf.nn.leaky_relu(conv7a, alpha=0.01)


        conv7b = tf.nn.conv3d(conv7a, weights['conv7b'], strides=[1, 1, 1, 1, 1], padding="SAME")
        conv7b = tf.add(conv7b, biases['conv7b'])
        conv7b = tf.nn.leaky_relu(conv7b, alpha=0.01)

        conv7b = tf.add(unpool7,conv7b)

        unpool8 = unpool(conv7b)

        conv8a = tf.nn.conv3d(unpool8, weights['conv8a'], strides=[1, 1, 1, 1, 1], padding="SAME")
        conv8a = tf.add(conv8a, biases['conv8a'])
        conv8a = tf.nn.leaky_relu(conv8a, alpha=0.01)


        conv8b = tf.nn.conv3d(conv8a, weights['conv8b'], strides=[1, 1, 1, 1, 1], padding="SAME")
        conv8b = tf.add(conv8b, biases['conv8b'])
        conv8b = tf.nn.leaky_relu(conv8b, alpha=0.01)

        conv8b = tf.add(conv8b,unpool8)

        unpool9 = unpool(conv8b)


        conv9a = tf.nn.conv3d(unpool9, weights['conv9a'], strides=[1, 1, 1, 1, 1], padding="SAME")
        conv9a = tf.add(conv9a, biases['conv9a'])
        conv9a = tf.nn.leaky_relu(conv9a, alpha=0.01)

        conv9b = tf.nn.conv3d(conv9a, weights['conv9b'], strides=[1, 1, 1, 1, 1], padding="SAME")
        conv9b = tf.add(conv9b, biases['conv9b'])
        conv9b = tf.nn.leaky_relu(conv9b, alpha=0.01)


        conv9c = tf.nn.conv3d(unpool9, weights['conv9c'], strides=[1, 1, 1, 1, 1], padding="SAME")
        conv9c = tf.add(conv9c, biases['conv9c'])
        conv9c = tf.nn.leaky_relu(conv9c, alpha=0.01)


        conv9c = tf.add(conv9c,conv9b)

        conv10a = tf.nn.conv3d(conv9c, weights['conv10a'], strides=[1, 1, 1, 1, 1], padding="SAME")
        conv10a = tf.add(conv10a, biases['conv10a'])
        conv10a = tf.nn.leaky_relu(conv10a, alpha=0.01)


        conv10b = tf.nn.conv3d(conv10a, weights['conv10b'], strides=[1, 1, 1, 1, 1], padding="SAME")
        conv10b = tf.add(conv10b, biases['conv10b'])
        conv10b = tf.nn.leaky_relu(conv10b, alpha=0.01)

        conv10c = tf.nn.conv3d(conv10b, weights['conv10c'], strides=[1, 1, 1, 1, 1], padding="SAME")
        conv10c = tf.add(conv10c, biases['conv10c'])       
        conv10c = tf.nn.leaky_relu(conv10c, alpha=0.01)     

        conv11a = tf.nn.conv3d(conv10c, weights['conv11a'], strides=[1, 1, 1, 1, 1], padding="SAME")
        conv11a = tf.add(conv11a, biases['conv11a'])

        exp_x = tf.exp(conv11a)  
        sum_exp_x = tf.reduce_sum(exp_x, axis=4, keepdims=True) 

        loss = tf.reduce_mean(
            tf.reduce_sum(-Y * conv11a, axis=4, keepdims=True) +
            tf.log(sum_exp_x)
        )

    return conv11a, loss


# Construct model
gru_op = gru()
output, loss = decoder()

optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    
    sess.run(init)

    merged = tf.summary.merge_all()

    train_writer = tf.summary.FileWriter('./train', sess.graph)

    x_train = dataset.train_data()
    y_train = dataset.train_labels()

    i = 0

    prev_state = np.zeros([n_gru_vox, n_gru_vox, n_gru_vox, 1, n_deconvfilter[0]])

    while (i < num_steps):

        for image_hash in x_train.keys():

            i += 1
            images = x_train[image_hash]
            shuffle(images)  # Shuffle views
            for image in images:
                ims = tf.convert_to_tensor(image)
                ims = tf.reshape(ims, [-1, 127, 127, 3])
                ims = ims.eval()

                prev_s = sess.run([gru_op], feed_dict={X: ims, p_H: prev_state})
                prev_s = np.array(prev_s)
                prev_s = prev_s[0, :, :, :, :, :]
                prev_state = prev_s

            vox = tf.convert_to_tensor(y_train[image_hash])
            vox = tf.cast(vox, tf.int64)
            vox = vox.eval()

            batch_voxel = np.zeros((32, 32, 32, 1, 2), dtype=int)

            batch_voxel[:, :, :, 0, 1] = vox
            batch_voxel[:, :, :, 0, 0] = vox < 1

            # Run optimization op (backprop) and cost op (to get loss value)
            l, o, _ = sess.run([loss, output, optimizer], feed_dict={G: prev_state, Y: batch_voxel})

            # train_writer.add_summary(summary, i)

            if (i % 2 == 0):
                print("Creating prediction objects.")

                exp_x = tf.exp(o)  # 32, 32, 32, 1 ,2
                sum_exp_x = tf.reduce_sum(exp_x, axis=4, keepdims=True)  # 32, 32, 32, 1, 1

                pred = exp_x / sum_exp_x

                pred = pred.eval()

                pred_name = "test_pred_" + str(i) + ".obj"
                pred_name2 = "test_pred_" + str(i) + "_XD.obj"

                voxel.voxel2obj(pred_name2, pred[:, :, :, 0, 0] > [0.4])
                voxel.voxel2obj(pred_name, pred[:, :, :, 0, 1] > [0.4])

            # Display logs per step
            print('Step %i: Loss: %f' % (i, l))
