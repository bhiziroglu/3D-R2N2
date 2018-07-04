import numpy as np
import argparse
import pprint
import logging
import multiprocessing as mp
import os
import tensorflow as tf
import net
import dataset
import voxel

#tf.logging.set_verbosity(tf.logging.INFO)
n_deconvfilter = [128, 128, 128, 64, 32, 2]
n_gru_vox = 4

def main():
    
    parser = argparse.ArgumentParser(description='Make everything 3D')
    parser.add_argument('--batch-size',dest='batch_size',help='Batch size',default=120,type=int)
    parser.add_argument('--iter',dest='iter',help='Number of iterations',default=1000,type=int)
    parser.add_argument('--weights', dest='weights', help='Pre-trained weights', default=None)
    args = parser.parse_args()
    print('Called with args:' , args)


    #w = net.initialize_weights()

    with tf.name_scope("Dataset"):
        x_train = dataset.train_data()
        y_train = dataset.train_labels()
    print("Finished reading dataset.")


    # TF Graph Input
    #X = tf.placeholder(tf.float32, shape=[1, 127, 127, 3],name = "Image")
    #Y = tf.placeholder(tf.float32, shape=[32, 32, 32],name = "Pred")
    #S = tf.placeholder(tf.float32, shape=[1,n_gru_vox,n_deconvfilter[0],n_gru_vox,n_gru_vox],name = "Hidden_State")

    #initial_state = tf.Variable(tf.zeros_like(
    #    tf.truncated_normal([1,n_gru_vox,n_deconvfilter[0],n_gru_vox,n_gru_vox], stddev=0.5)), name="initial_state")
 

    forward_pass = net.encoder_gru()

    decoder_pass = net.decoder()

    logits = decoder_pass

    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    #loss_op = net.loss(logits,Y)

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
        print("Started training.")
        for image_hash in x_train.keys():
            iter+=1

            initial_state = tf.zeros_like(
              tf.truncated_normal([1,n_gru_vox,n_deconvfilter[0],n_gru_vox,n_gru_vox], stddev=0.5))
            initial_state = initial_state.eval()

            for image in x_train[image_hash]:
                image = tf.convert_to_tensor(image)
                image = tf.reshape(image,[1,127,127,3])
                image = image.eval()
                #print("XDXDXD")
                #print(initial_state.shape)
                initial_state = sess.run([forward_pass], feed_dict={X: image, S: initial_state})
                #initial_state = tf.convert_to_tensor(hidden_state)
                #initial_state = initial_state.eval()
            


            vox = tf.convert_to_tensor(y_train[image_hash])
            vox = vox.eval()

            loss, _ = sess.run([loss_op, update_step], feed_dict={S: initial_state, Y: vox})
            
            print("Image: ",iter," LOSS:  ",loss)
            tf.summary.histogram('loss', loss)

            if iter % 2 == 0:
                print("Testing Model at Iter ",iter)
                # Save the prediction to an OBJ file (mesh file).
                net.predict(w,"test_image.png",iter)
                del x_train
                del y_train
                del args
                del w
                print("Finished early!")
                return


        print("Finished!")



    del x_train
    del y_train
    del args
    del w






if __name__ == '__main__':
    main()