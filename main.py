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
    loss_op, _ = net.loss(logits,Y)
    #optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
    #train_op = optimizer.minimize(loss_op)

    # Calculate and clip gradients
    params = tf.trainable_variables()
    gradients = tf.gradients(loss_op, params)
    clipped_gradients, _ = tf.clip_by_global_norm(
        gradients, 1) # 1 is max_gradient_norm

    # Optimization
    optimizer = tf.train.AdamOptimizer(0.00001)
    update_step = optimizer.apply_gradients(
        zip(clipped_gradients, params))

    # Initialize the variables
    init = tf.initialize_all_variables()



    # Start training
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./train',
                                            sess.graph)
                                            
        iter = 0

        for images in x_train.keys():
            iter+=1
            # Run optimization op (backprop)
            summary = sess.run([loss_op, update_step], feed_dict={X: x_train[images], Y: y_train[images]})
            train_writer.add_summary(summary, iter)
            if iter % 10 == 0:
                # Calculate batch loss and accuracy
                loss = sess.run([loss_op, update_step], feed_dict={X: x_train[images],
                                                                    Y: y_train[images]})
                
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