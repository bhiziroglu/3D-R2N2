import numpy as np
import argparse
import pprint
import logging
import multiprocessing as mp
import os
import tensorflow as tf
import net
import dataset

#tf.logging.set_verbosity(tf.logging.INFO)
n_convfilter = [96, 128, 256, 256, 256, 256]
n_deconvfilter = [128, 128, 128, 64, 32, 2]
n_gru_vox = 4
n_fc_filters = [1024]

def build_graph():
    
    ## ENCODER PART ##
    with tf.name_scope("Encoder"):
        w0 = tf.Variable(tf.contrib.layers.xavier_initializer()((7,7,3,n_convfilter[0])), name="conv1a")
        w1 = tf.Variable(tf.contrib.layers.xavier_initializer()((3,3,n_convfilter[0],n_convfilter[0])), name="conv1b")
        w2 = tf.Variable(tf.contrib.layers.xavier_initializer()((3,3,n_convfilter[0],n_convfilter[1])), name="conv2a")
        w3 = tf.Variable(tf.contrib.layers.xavier_initializer()((3,3,n_convfilter[1],n_convfilter[1])), name="conv2b")
        w4 = tf.Variable(tf.contrib.layers.xavier_initializer()((1,1,n_convfilter[0],n_convfilter[1])), name="conv2c")
        w5 = tf.Variable(tf.contrib.layers.xavier_initializer()((3,3,n_convfilter[1],n_convfilter[2])), name="conv3a")
        w6 = tf.Variable(tf.contrib.layers.xavier_initializer()((3,3,n_convfilter[2],n_convfilter[2])), name="conv3b")
        w7 = tf.Variable(tf.contrib.layers.xavier_initializer()((1,1,n_convfilter[1],n_convfilter[2])), name="conv3c")
        w8 = tf.Variable(tf.contrib.layers.xavier_initializer()((3,3,n_convfilter[2],n_convfilter[3])), name="conv4a")
        w9 = tf.Variable(tf.contrib.layers.xavier_initializer()((3,3,n_convfilter[3],n_convfilter[3])), name="conv4b")
        w10 = tf.Variable(tf.contrib.layers.xavier_initializer()((3,3,n_convfilter[3],n_convfilter[4])), name="conv5a")
        w11 = tf.Variable(tf.contrib.layers.xavier_initializer()((3,3,n_convfilter[4],n_convfilter[4])), name="conv5b")
        w12 = tf.Variable(tf.contrib.layers.xavier_initializer()((1,1,n_convfilter[4],n_convfilter[4])), name="conv5c")
        w13 = tf.Variable(tf.contrib.layers.xavier_initializer()((3,3,n_convfilter[4],n_convfilter[5])), name="conv6a")
        w14 = tf.Variable(tf.contrib.layers.xavier_initializer()((3,3,n_convfilter[5],n_convfilter[5])), name="conv6b")
        w15 = tf.Variable(tf.contrib.layers.xavier_initializer()((1,n_fc_filters[0])), name="fc7")
    
    ## GRU PART ##
    with tf.name_scope("GRU"):
        w16 = tf.Variable(tf.contrib.layers.xavier_initializer()((1024,8192)), name="w_update")
        w17 = tf.Variable(tf.contrib.layers.xavier_initializer()((3,3,3,n_gru_vox,n_gru_vox)), name="update_gate")
        w18 = tf.Variable(tf.contrib.layers.xavier_initializer()((3,3,3,n_gru_vox,n_gru_vox)), name="reset_gate")
        w19 = tf.Variable(tf.contrib.layers.xavier_initializer()((3,3,3,n_gru_vox,n_gru_vox)), name="tanh_reset")
        w31 = tf.Variable(tf.contrib.layers.xavier_initializer()((1024,8192)), name="w_reset")
        
    ## DECODER PART ##
    with tf.name_scope("Decoder"):
        w20 = tf.Variable(tf.contrib.layers.xavier_initializer()((3,3,3,n_deconvfilter[1],n_deconvfilter[1])), name="conv7a")
        w21 = tf.Variable(tf.contrib.layers.xavier_initializer()((3,3,3,n_deconvfilter[1],n_deconvfilter[1])), name="conv7b")
        w22 = tf.Variable(tf.contrib.layers.xavier_initializer()((3,3,3,n_deconvfilter[1],n_deconvfilter[2])), name="conv8a")
        w23 = tf.Variable(tf.contrib.layers.xavier_initializer()((3,3,3,n_deconvfilter[2],n_deconvfilter[2])), name="conv8b")
        w24 = tf.Variable(tf.contrib.layers.xavier_initializer()((3,3,3,n_deconvfilter[2],n_deconvfilter[3])), name="conv9a")
        w25 = tf.Variable(tf.contrib.layers.xavier_initializer()((3,3,3,n_deconvfilter[3],n_deconvfilter[3])), name="conv9b")
        w26 = tf.Variable(tf.contrib.layers.xavier_initializer()((1,1,1,n_deconvfilter[2],n_deconvfilter[3])), name="conv9c")
        w27 = tf.Variable(tf.contrib.layers.xavier_initializer()((3,3,3,n_deconvfilter[3],n_deconvfilter[4])), name="conv10a")
        w28 = tf.Variable(tf.contrib.layers.xavier_initializer()((3,3,3,n_deconvfilter[4],n_deconvfilter[4])), name="conv10b")
        w29 = tf.Variable(tf.contrib.layers.xavier_initializer()((3,3,3,n_deconvfilter[4],n_deconvfilter[4])), name="conv10c")
        w30 = tf.Variable(tf.contrib.layers.xavier_initializer()((3,3,3,n_deconvfilter[4],n_deconvfilter[5])), name="conv11a")






def main():
    
    parser = argparse.ArgumentParser(description='Make everything 3D')
    parser.add_argument('--batch-size',dest='batch_size',help='Batch size',default=120,type=int)
    parser.add_argument('--iter',dest='iter',help='Number of iterations',default=1000,type=int)
    parser.add_argument('--weights', dest='weights', help='Pre-trained weights', default=None)
    args = parser.parse_args()
    print('Called with args:' , args)


    w = net.initialize_weights()
    x_train = dataset.train_data()
    y_train = dataset.train_labels()
    #net.train(w,x_train,y_train) # Train the network


    # TF Graph Input
    X = tf.placeholder(tf.float32, shape=[24, 127, 127, 3],name = "Image")
    Y = tf.placeholder(tf.float32, shape=[32, 32, 32],name = "Pred")
    #Yhat = tf.placeholder(tf.float32, shape=[32, 32, 32],name = "Ground Truth")


    initial_state = tf.Variable(tf.zeros_like(
        tf.truncated_normal([1,n_gru_vox,n_deconvfilter[0],n_gru_vox,n_gru_vox], stddev=0.5)), name="initial_state")
 

    enc = net.encoder(w,X)
    gru = net.gru(w,enc,initial_state)
    dec = net.decoder(w,gru)

    logits = dec

    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = net.loss(logits,Y)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
    train_op = optimizer.minimize(loss_op)

    # Calculate and clip gradients
    #params = tf.trainable_variables()
    #gradients = tf.gradients(loss_op, params)
    #clipped_gradients, _ = tf.clip_by_global_norm(
    #    gradients, 1) # 1 is max_gradient_norm

    # Optimization
    #optimizer = tf.train.AdamOptimizer(0.00001)
    #update_step = optimizer.apply_gradients(
    #    zip(clipped_gradients, params))

    # Initialize the variables
    init = tf.global_variables_initializer()



    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./train',
                                            sess.graph)
                                            
        iter = 0

        for image_hash in x_train.keys():
            iter+=1
            ims = []
            for image in x_train[image_hash]:
                ims.append(image)
            
            #ims = np.asarray(ims)
            #vox = np.asarray(y_train[image_hash])
            # Run optimization op (backprop)

            ims = tf.Session().run(tf.convert_to_tensor(ims)) # Convert List->Tensor->Numpy Array
            vox = tf.Session().run(tf.convert_to_tensor(y_train[image_hash])) # Convert List->Tensor->Numpy Array

            sess.run([train_op], feed_dict={X: ims, Y: vox})
            #train_writer.add_summary(summary, iter)
            if iter % 10 == 0:
                # Calculate batch loss and accuracy
                loss, _ = sess.run(train_op, feed_dict={X: ims,
                                                        Y: vox})
                
                print("Step " + str(iter) + " Loss= " + loss)

        print("Optimization Finished!")


        # Calculate accuracy for 128 mnist test images
        #test_len = 128
        #test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
        #test_label = mnist.test.labels[:test_len]
        #print("Testing Accuracy:", \
        #    sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))




    del x_train
    del y_train
    del args
    del w






if __name__ == '__main__':
    main()