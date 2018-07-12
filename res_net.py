import tensorflow as tf
import voxel
import numpy as np
from PIL import Image
import dataset
from tqdm import tqdm
#from tensorflow.python import debug as tf_debug


n_convfilter = [96, 128, 256, 256, 256, 256]
n_deconvfilter = [128, 128, 128, 64, 32, 2]
n_gru_vox = 4   
n_fc_filters = [1024]
NUM_OF_IMAGES = 24

def initialize_placeholders():
    with tf.name_scope("Placeholders"):
        X = tf.placeholder(tf.float32, shape=[NUM_OF_IMAGES, 127, 127, 3],name = "X")
        Y = tf.placeholder(tf.float32, shape=[1, 32, 32, 32, 2],name = "Y")
    return X,Y

X,Y = initialize_placeholders()

encoder_gru_kernel_shapes = [
    #Encoder
    [7,7,3,n_convfilter[0]], #conv1a
    [3,3,n_convfilter[0],n_convfilter[0]], #conv1b
    [3,3,n_convfilter[0],n_convfilter[1]], #conv2a
    [3,3,n_convfilter[1],n_convfilter[1]], #conv2b
    [1,1,n_convfilter[0],n_convfilter[1]], #conv2c
    [3,3,n_convfilter[1],n_convfilter[2]], #conv3a
    [3,3,n_convfilter[2],n_convfilter[2]], #conv3b
    [1,1,n_convfilter[1],n_convfilter[2]], #conv3c
    [3,3,n_convfilter[2],n_convfilter[3]], #conv4a
    [3,3,n_convfilter[3],n_convfilter[3]], #conv4b
    [3,3,n_convfilter[3],n_convfilter[4]], #conv5a
    [3,3,n_convfilter[4],n_convfilter[4]], #conv5b
    [1,1,n_convfilter[4],n_convfilter[4]], #conv5c
    [3,3,n_convfilter[4],n_convfilter[5]], #conv6a
    [3,3,n_convfilter[5],n_convfilter[5]], #conv6b
    [1,n_fc_filters[0]], #fc7
    #GRU
    [1024,8192], #w_update
    [3,3,3,n_gru_vox,n_gru_vox], #update_gate
    [3,3,3,n_gru_vox,n_gru_vox], #reset_gate
    [3,3,3,n_gru_vox,n_gru_vox], #tanh_reset
    [1024,8192] #w_reset
]

decoder_kernel_shapes = [
    [3,3,3,n_deconvfilter[1],n_deconvfilter[1]], #conv7a #0
    [3,3,3,n_deconvfilter[1],n_deconvfilter[1]], #conv7b #1
    [3,3,3,n_deconvfilter[1],n_deconvfilter[2]], #conv8a #2
    [3,3,3,n_deconvfilter[2],n_deconvfilter[2]], #conv8b #3
    [3,3,3,n_deconvfilter[2],n_deconvfilter[3]], #conv9a #4
    [3,3,3,n_deconvfilter[3],n_deconvfilter[3]], #conv9b #5
    [1,1,1,n_deconvfilter[2],n_deconvfilter[3]], #conv9c #6
    [3,3,3,n_deconvfilter[3],n_deconvfilter[4]], #conv10a#7
    [3,3,3,n_deconvfilter[4],n_deconvfilter[4]], #conv10b#8
    [3,3,3,n_deconvfilter[4],n_deconvfilter[4]], #conv10c#9
    [64,32],                                     #fc
    [3,3,3,n_deconvfilter[4],n_deconvfilter[5]]  #conv11a #10
]


def build_graph():
    
    # Init weights

    with tf.name_scope("encoder_gru_weights"):
        w = [tf.get_variable(
            "w"+str(_), shape=kernel, initializer = tf.glorot_normal_initializer(seed=4444), trainable=True) 
            for _,kernel in enumerate(encoder_gru_kernel_shapes)]

    with tf.name_scope("decoder_weights"):
        w_decoder = [tf.get_variable(
            "w2_"+str(_), shape=kernel, initializer = tf.glorot_normal_initializer(seed=4321), trainable=True) 
            for _,kernel in enumerate(decoder_kernel_shapes)]

    with tf.name_scope("Encoder"):
        
        # Convolutional Layer #1
        conv1a = tf.nn.conv2d(input=X,filter=w[0],strides=[1,1,1,1],padding="SAME")
        conv1a = tf.nn.leaky_relu(conv1a,alpha=0.01)
        conv1b = tf.nn.conv2d(input=conv1a,filter=w[1],strides=[1,1,1,1],padding="SAME")
        conv1b = tf.nn.leaky_relu(conv1b,alpha=0.01)
        pool1 = tf.nn.max_pool(conv1b,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        # [1, 64, 64, 96]

        # Convolutional Layer #2
        conv2a = tf.nn.conv2d(input=pool1,filter=w[2],strides=[1,1,1,1],padding="SAME")
        conv2a = tf.nn.leaky_relu(conv2a,alpha=0.01)
        conv2b = tf.nn.conv2d(input=conv2a,filter=w[3],strides=[1,1,1,1],padding="SAME")
        conv2b = tf.nn.leaky_relu(conv2b,alpha=0.01)
        conv2c = tf.nn.conv2d(input=pool1,filter=w[4],strides=[1,1,1,1],padding="SAME")
        conv2c = tf.nn.leaky_relu(conv2c,alpha=0.01)
        res2 = tf.add(conv2b,conv2c)
        pool2 = tf.nn.max_pool(res2,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        ''' !!!TODO!!!  (1, 32, 32, 128)   ->>>      Paper result size is (1, 33, 33, 128)'''

        # Convolutional Layer #3
        conv3a = tf.nn.conv2d(input=pool2,filter=w[5],strides=[1,1,1,1],padding="SAME")
        conv3a = tf.nn.leaky_relu(conv3a,alpha=0.01)
        conv3b = tf.nn.conv2d(input=conv3a,filter=w[6],strides=[1,1,1,1],padding="SAME")
        conv3b = tf.nn.leaky_relu(conv3b,alpha=0.01)
        conv3c = tf.nn.conv2d(input=pool2,filter=w[7],strides=[1,1,1,1],padding="SAME")
        conv3c = tf.nn.leaky_relu(conv3c,alpha=0.01)
        res3 = tf.add(conv3b,conv3c)
        pool3 = tf.nn.max_pool(res3,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        ''' !!!TODO!!!  (1, 16, 16, 256)   ->>>      Paper result size is (1, 17, 17, 256)'''

        # Convolutional Layer #4
        conv4a = tf.nn.conv2d(input=pool3,filter=w[8],strides=[1,1,1,1],padding="SAME")
        conv4a = tf.nn.leaky_relu(conv4a,alpha=0.01)
        conv4b = tf.nn.conv2d(input=conv4a,filter=w[9],strides=[1,1,1,1],padding="SAME")
        conv4b = tf.nn.leaky_relu(conv4b,alpha=0.01)
        pool4 = tf.nn.max_pool(conv4b,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        ''' !!!TODO!!!  (1, 8, 8, 256)   ->>>      Paper result size is (1, 9, 9, 256)'''
    
        # Convolutional Layer #5
        conv5a = tf.nn.conv2d(input=pool4,filter=w[10],strides=[1,1,1,1],padding="SAME")
        conv5a = tf.nn.leaky_relu(conv5a,alpha=0.01)
        conv5b = tf.nn.conv2d(input=conv5a,filter=w[11],strides=[1,1,1,1],padding="SAME")
        conv5b = tf.nn.leaky_relu(conv5b,alpha=0.01)
        conv5c = tf.nn.conv2d(input=pool4,filter=w[12],strides=[1,1,1,1],padding="SAME")
        conv5c = tf.nn.leaky_relu(conv5c,alpha=0.01)
        res5 = tf.add(conv5b,conv5c)
        pool5 = tf.nn.max_pool(res5,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        ''' !!!TODO!!!  (1, 4, 4, 256)   ->>>      Paper result size is (1, 5, 5, 256)'''
    
        # Convolutional Layer #6
        conv6a = tf.nn.conv2d(input=pool5,filter=w[13],strides=[1,1,1,1],padding="SAME")
        conv6a = tf.nn.leaky_relu(conv6a,alpha=0.01)
        conv6b = tf.nn.conv2d(input=conv6a,filter=w[14],strides=[1,1,1,1],padding="SAME")
        conv6b = tf.nn.leaky_relu(conv6b,alpha=0.01)
        pool6 = tf.nn.max_pool(conv6b,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        ''' !!!TODO!!!  (1, 2, 2, 256)   ->>>      Paper result size is (1, 3, 3, 256)'''
        
        # Flatten Layer
        flat7 = tf.reshape(pool6,[pool6.shape[0],-1])
        ''' !!!TODO!!!  (1, 1024)   ->>>      Paper result size is (1, 2304)'''

        # FC Layer
        fc7 = tf.multiply(flat7,w[15])
        ''' w[15] was [1024] , now its [1,1024]. Which one is correct?'''
        # [N,1024]

    
    prev_s = tf.truncated_normal([1,n_gru_vox,n_deconvfilter[0],n_gru_vox,n_gru_vox], stddev=0.5)
            


    with tf.name_scope("GRU"):            
    
        for index in range(fc7.shape[0]):
            
            curr_image = fc7[index,:] # Take single image out
            curr_image = tf.reshape(curr_image,[1,1024])
            
            fc_layer_U = tf.matmul(curr_image,w[16]) #[1,1024]x[1024,8192] // FC LAYER FOR UPDATE GATE
            fc_layer_U = tf.reshape(fc_layer_U,[-1,4,128,4,4]) #[1,4,128,4,4]

            fc_layer_R = tf.matmul(curr_image,w[20]) # FC LAYER FOR RESET GATE
            fc_layer_R = tf.reshape(fc_layer_R,[-1,4,128,4,4]) #[1,4,128,4,4]
            
            update_gate = tf.nn.conv3d(prev_s,w[17],strides=[1,1,1,1,1],padding="SAME")
            #update_gate = tf.nn.leaky_relu(update_gate,alpha=0.01)

            update_gate = update_gate + fc_layer_U
            update_gate = tf.sigmoid(update_gate)

            complement_update_gate = tf.subtract(tf.ones_like(update_gate),update_gate)

            reset_gate = tf.nn.conv3d(prev_s,w[18],strides=[1,1,1,1,1],padding="SAME")
            #reset_gate = tf.nn.leaky_relu(reset_gate,alpha=0.01)
            reset_gate = reset_gate + fc_layer_R
            reset_gate = tf.sigmoid(reset_gate)

            rs = tf.multiply(reset_gate,prev_s) # Element-wise multiply

            tanh_reset = tf.nn.conv3d(rs,w[19],strides=[1,1,1,1,1],padding="SAME")
            #tanh_reset = tf.nn.leaky_relu(tanh_reset,alpha=0.01)
            tanh_reset = tf.tanh(tanh_reset)

            gru_out = tf.add(
                tf.multiply(update_gate,prev_s),
                tf.multiply(complement_update_gate,tanh_reset)
            )

            prev_s = gru_out
    
    with tf.name_scope("Decoder"):

        s = tf.transpose(prev_s,perm=[0,1,4,3,2]) # [(1, 4, 128, 4, 4)] -> [(1, 4, 4, 4, 128)]
        unpool7 = unpool(s)

        conv7a = tf.nn.conv3d(unpool7,w_decoder[0],strides=[1,1,1,1,1],padding="SAME")
        conv7a = tf.nn.leaky_relu(conv7a,alpha=0.01)

        conv7b = tf.nn.conv3d(conv7a,w_decoder[1],strides=[1,1,1,1,1],padding="SAME")
        conv7b = tf.nn.leaky_relu(conv7b,alpha=0.01)
        res7 = tf.add(unpool7,conv7b)

        unpool8 = unpool(res7)

        conv8a = tf.nn.conv3d(unpool8,w_decoder[2],strides=[1,1,1,1,1],padding="SAME")
        conv8a = tf.nn.leaky_relu(conv8a,alpha=0.01)   

        conv8b = tf.nn.conv3d(conv8a,w_decoder[3],strides=[1,1,1,1,1],padding="SAME")
        conv8b = tf.nn.leaky_relu(conv8b,alpha=0.01)    
        res8 = tf.add(unpool8,conv8b)

        unpool9 = unpool(res8)

        conv9a = tf.nn.conv3d(unpool9,w_decoder[4],strides=[1,1,1,1,1],padding="SAME")
        conv9a = tf.nn.leaky_relu(conv9a,alpha=0.01)   

        conv9b = tf.nn.conv3d(conv9a,w_decoder[5],strides=[1,1,1,1,1],padding="SAME")
        conv9b = tf.nn.leaky_relu(conv9b,alpha=0.01)  

        conv9c = tf.nn.conv3d(unpool9,w_decoder[6],strides=[1,1,1,1,1],padding="SAME")

        res9 = tf.add(conv9c,conv9b)
        unpool10 = res9

        conv10a = tf.nn.conv3d(unpool10,w_decoder[7],strides=[1,1,1,1,1],padding="SAME")
        conv10a = tf.nn.leaky_relu(conv10a,alpha=0.01)  
        
        conv10b = tf.nn.conv3d(conv10a,w_decoder[8],strides=[1,1,1,1,1],padding="SAME")
        conv10b = tf.nn.leaky_relu(conv10b,alpha=0.01)  

        conv10c = tf.nn.conv3d(conv10b,w_decoder[9],strides=[1,1,1,1,1],padding="SAME")
        conv10c = tf.nn.leaky_relu(conv10c,alpha=0.01)  

        unpool10_ = tf.matmul(tf.reshape(unpool10,[-1,64]),w_decoder[10])
        unpool10_ = tf.reshape(unpool10_,[1,32,32,32,32])

        res10 = tf.add(conv10c,unpool10_)

        conv11a = tf.nn.conv3d(res10,w_decoder[11],strides=[1,1,1,1,1],padding="SAME")
        conv11a = tf.nn.leaky_relu(conv11a,alpha=0.01)  
        
        #conv11a = tf.contrib.layers.layer_norm(conv11a) #Norm layer

    #conv11a = tf.reduce_max(conv11a,axis=4,keepdims=True)
    exp_x = tf.exp(conv11a)
    sum_exp_x = tf.reduce_sum(exp_x,axis=4,keepdims=True)

    #loss_ = paper_loss(conv11a,sum_exp_x,Y) #conv11a is final prediction. Y is ground truth.
    loss_ = loss(conv11a,sum_exp_x,Y)
    pred = exp_x / sum_exp_x
    return loss_, pred

 

def loss(y,sum_exp_x,yhat):
    return tf.reduce_mean(
        tf.reduce_sum(-yhat * y,axis=4,keepdims=True)+tf.log(sum_exp_x)
    )


def test_predict(pred,ind):
    pred_name = "test_pred_"+str(ind)+".obj"
    voxel.voxel2obj(pred_name,pred)


def unpool_2d_zero_filled(x):
    out = tf.concat([x, tf.zeros_like(x)], 3)
    out = tf.concat([out, tf.zeros_like(out)], 2)

    sh = x.get_shape().as_list()
    out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
    return tf.reshape(out, out_size)

def unpool(x): #unpool_3d_zero_filled
    # https://github.com/tensorflow/tensorflow/issues/2169
    out = tf.concat([x, tf.zeros_like(x)], 4)
    out = tf.concat([out, tf.zeros_like(out)], 3)
    out = tf.concat([out, tf.zeros_like(out)], 2)

    sh = x.get_shape().as_list()
    out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3] * 2, sh[4]]
    return tf.reshape(out, out_size)

loss_, pred_ = build_graph()

# Optimization
optimizer = tf.train.AdamOptimizer(1e-4)
grads_vars = optimizer.compute_gradients(loss_)
clipped_gradients = []
for grad,var in grads_vars:
    clipped_gradients.append(
        (tf.clip_by_value(grad,5.0,-5.0),var)) # 1 is max_gradient_norm)

updates = optimizer.apply_gradients(clipped_gradients)


if __name__=="__main__":
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:


        #sess = tf_debug.TensorBoardDebugWrapperSession(sess, "Berkan-MacBook-Pro.local:4334")
        # Run the initializer
        sess.run(init)

        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter('./train',
                                            sess.graph)
                                            
        iter = 0
        print("Started training.")

        with open("log.txt", "w") as myfile:
            myfile.write("3D-R2N2 Started training.\n")

        x_train = dataset.train_data()
        y_train = dataset.train_labels()

        #pbar = tqdm(total=dataset.TOTAL_SIZE)
        pbar = tqdm(total=10)
        #while(x_train!=[] and y_train!=[]):
        while(iter<10): # 5000 iterations
            for image_hash in x_train.keys():
                iter+=1

                initial_state = tf.truncated_normal([1,n_gru_vox,n_deconvfilter[0],n_gru_vox,n_gru_vox], stddev=0.5)
                initial_state = initial_state.eval()

                ims = []

                for image in x_train[image_hash]:
                    ims.append(image)
                
                ims = tf.convert_to_tensor(ims)
                ims = tf.reshape(ims,[-1,127,127,3])
                ims = ims.eval()
                #ims = np.random.rand(24,127,127,3)

                vox = tf.convert_to_tensor(y_train[image_hash])
                vox = tf.cast(vox,tf.float32)
                vox = vox.eval()
                batch_voxel = np.zeros((1,32,32,32,2), dtype=float)  
                batch_voxel[0,:,:,:,0]= vox < 1
                batch_voxel[0,:,:,:,1]= vox

                l,u,o = sess.run([loss_, updates, pred_], feed_dict={X: ims, Y: batch_voxel})

                print("OBJECT: " + str(iter)+" LOSS: "+str(l))
                with open("log.txt", "a") as myfile:
                    myfile.write("Iteration: "+str(iter)+" Loss: "+str(l)+"\n")
                #tf.summary.histogram('loss', forw[0])
                if iter % 5 == 0:
                    print("Testing Model at Iter ",iter)
                    print("HASH "+image_hash)
                    # Save the prediction to an OBJ file (mesh file).
                    #predict(w,"test_image.png",iter)
                    test_predict(o[0,:,:,:,0],iter)
                    #test_predict(vox,iter+10)
                
                pbar.update(1)
            
            


                    
            #x_train = dataset.train_data()
            #y_train = dataset.train_labels()

        pbar.close()
        print("Finished!")

