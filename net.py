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
        #Y = tf.placeholder(tf.float32, shape=[32, 32, 32, 1, 2],name = "Y")
        Y = tf.placeholder(tf.float32, shape=[24,1024],name = "Y")
    return X,Y

X,Y = initialize_placeholders()

encoder_gru_kernel_shapes = [
    #Encoder
    [7,7,3,n_convfilter[0]], #conv1a
    [3,3,n_convfilter[0],n_convfilter[1]], #conv2a
    [3,3,n_convfilter[1],n_convfilter[2]], #conv3a
    [3,3,n_convfilter[2],n_convfilter[3]], #conv4a
    [3,3,n_convfilter[3],n_convfilter[4]], #conv5a
    [3,3,n_convfilter[4],n_convfilter[5]], #conv6a
    [1,n_fc_filters[0]], #fc7 #6
    #GRU
    [1024,8192], #w_update and w_reset
    [3,3,3,n_deconvfilter[0],n_deconvfilter[0]], #update_gate
    [3,3,3,n_deconvfilter[0],n_deconvfilter[0]], #reset_gate
    [3,3,3,n_deconvfilter[0],n_deconvfilter[0]] #tanh_reset
]

decoder_kernel_shapes = [
    [3,3,3,n_deconvfilter[0],n_deconvfilter[1]], #conv7a #0
    [3,3,3,n_deconvfilter[1],n_deconvfilter[2]], #conv8a #2
    [3,3,3,n_deconvfilter[2],n_deconvfilter[3]], #conv9a #4
    [3,3,3,n_deconvfilter[3],n_deconvfilter[4]], #conv10a#7
    [3,3,3,n_deconvfilter[4],n_deconvfilter[5]]  #conv11a #10
]


# Init weights

with tf.name_scope("encoder_gru_weights"):
    w = [tf.get_variable(
        "w"+str(_), shape=kernel, initializer = tf.glorot_normal_initializer(seed=3), trainable=True) 
        for _,kernel in enumerate(encoder_gru_kernel_shapes)]
    for w_ in w:
        tf.summary.histogram(w_.name, w_)

with tf.name_scope("decoder_weights"):
    w_decoder = [tf.get_variable(
        "w2_"+str(_), shape=kernel, initializer = tf.glorot_normal_initializer(seed=5), trainable=True) 
        for _,kernel in enumerate(decoder_kernel_shapes)]
    for w_ in w_decoder:
        tf.summary.histogram(w_.name, w_)

def try_encoder():
    
    with tf.name_scope("Encoder"):
        
        # Convolutional Layer #1
        conv1a = tf.nn.conv2d(input=X,filter=w[0],strides=[1,1,1,1],padding="SAME")
        conv1a = tf.nn.max_pool(conv1a,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv1a = tf.nn.leaky_relu(conv1a,alpha=0.01)
        # [1, 64, 64, 96]

        # Convolutional Layer #2
        conv2a = tf.nn.conv2d(input=conv1a,filter=w[1],strides=[1,1,1,1],padding="SAME")
        conv2a = tf.nn.max_pool(conv2a,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv2a = tf.nn.leaky_relu(conv2a,alpha=0.01)
        ''' !!!TODO!!!  (1, 32, 32, 128)   ->>>      Paper result size is (1, 33, 33, 128)'''

        # Convolutional Layer #3
        conv3a = tf.nn.conv2d(input=conv2a,filter=w[2],strides=[1,1,1,1],padding="SAME")
        conv3a = tf.nn.max_pool(conv3a,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv3a = tf.nn.leaky_relu(conv3a,alpha=0.01)
        ''' !!!TODO!!!  (1, 16, 16, 256)   ->>>      Paper result size is (1, 17, 17, 256)'''

        # Convolutional Layer #4
        conv4a = tf.nn.conv2d(input=conv3a,filter=w[3],strides=[1,1,1,1],padding="SAME")
        conv4a = tf.nn.max_pool(conv4a,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv4a = tf.nn.leaky_relu(conv4a,alpha=0.01)
        ''' !!!TODO!!!  (1, 8, 8, 256)   ->>>      Paper result size is (1, 9, 9, 256)'''
    
        # Convolutional Layer #5
        conv5a = tf.nn.conv2d(input=conv4a,filter=w[4],strides=[1,1,1,1],padding="SAME")
        conv5a = tf.nn.max_pool(conv5a,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv5a = tf.nn.leaky_relu(conv5a,alpha=0.01)
        ''' !!!TODO!!!  (1, 4, 4, 256)   ->>>      Paper result size is (1, 5, 5, 256)'''
    
        # Convolutional Layer #6
        conv6a = tf.nn.conv2d(input=conv5a,filter=w[5],strides=[1,1,1,1],padding="SAME")
        conv6a = tf.nn.max_pool(conv6a,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv6a = tf.nn.leaky_relu(conv6a,alpha=0.01)
        ''' !!!TODO!!!  (1, 2, 2, 256)   ->>>      Paper result size is (1, 3, 3, 256)'''
        
        # Flatten Layer
        flat7 = tf.reshape(conv6a,[conv6a.shape[0],-1])
        ''' !!!TODO!!!  (1, 1024)   ->>>      Paper result size is (1, 2304)'''

        # FC Layer
        fc7 = tf.multiply(flat7,w[6])
        ''' w[15] was [1024] , now its [1,1024]. Which one is correct?'''
        # [N,1024]


        loss_ = sample_loss(fc7)
        return loss_


def build_graph():


    with tf.name_scope("Encoder"):
        
        # Convolutional Layer #1
        conv1a = tf.nn.conv2d(input=X,filter=w[0],strides=[1,1,1,1],padding="SAME")
        conv1a = tf.nn.max_pool(conv1a,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv1a = tf.nn.leaky_relu(conv1a,alpha=0.01)
        # [1, 64, 64, 96]

        # Convolutional Layer #2
        conv2a = tf.nn.conv2d(input=conv1a,filter=w[1],strides=[1,1,1,1],padding="SAME")
        conv2a = tf.nn.max_pool(conv2a,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv2a = tf.nn.leaky_relu(conv2a,alpha=0.01)
        ''' !!!TODO!!!  (1, 32, 32, 128)   ->>>      Paper result size is (1, 33, 33, 128)'''

        # Convolutional Layer #3
        conv3a = tf.nn.conv2d(input=conv2a,filter=w[2],strides=[1,1,1,1],padding="SAME")
        conv3a = tf.nn.max_pool(conv3a,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv3a = tf.nn.leaky_relu(conv3a,alpha=0.01)
        ''' !!!TODO!!!  (1, 16, 16, 256)   ->>>      Paper result size is (1, 17, 17, 256)'''

        # Convolutional Layer #4
        conv4a = tf.nn.conv2d(input=conv3a,filter=w[3],strides=[1,1,1,1],padding="SAME")
        conv4a = tf.nn.max_pool(conv4a,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv4a = tf.nn.leaky_relu(conv4a,alpha=0.01)
        ''' !!!TODO!!!  (1, 8, 8, 256)   ->>>      Paper result size is (1, 9, 9, 256)'''
    
        # Convolutional Layer #5
        conv5a = tf.nn.conv2d(input=conv4a,filter=w[4],strides=[1,1,1,1],padding="SAME")
        conv5a = tf.nn.max_pool(conv5a,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv5a = tf.nn.leaky_relu(conv5a,alpha=0.01)
        ''' !!!TODO!!!  (1, 4, 4, 256)   ->>>      Paper result size is (1, 5, 5, 256)'''
    
        # Convolutional Layer #6
        conv6a = tf.nn.conv2d(input=conv5a,filter=w[5],strides=[1,1,1,1],padding="SAME")
        conv6a = tf.nn.max_pool(conv6a,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        conv6a = tf.nn.leaky_relu(conv6a,alpha=0.01)
        ''' !!!TODO!!!  (1, 2, 2, 256)   ->>>      Paper result size is (1, 3, 3, 256)'''
        
        # Flatten Layer
        flat7 = tf.reshape(conv6a,[conv6a.shape[0],-1])
        ''' !!!TODO!!!  (1, 1024)   ->>>      Paper result size is (1, 2304)'''

        # FC Layer
        fc7 = tf.multiply(flat7,w[6])
        ''' w[15] was [1024] , now its [1,1024]. Which one is correct?'''
        # [N,1024]
    
    prev_s = tf.truncated_normal([n_gru_vox,n_gru_vox,n_gru_vox,1,n_deconvfilter[0]], stddev=0.5)
            


    with tf.name_scope("GRU"):            
    
        for index in range(fc7.shape[0]):
            
            curr_image = fc7[index,:] # Take single image out
            curr_image = tf.reshape(curr_image,[1,1024])
            
            fc_layer = tf.matmul(curr_image,w[7]) #[1,1024]x[1024,8192] // FC LAYER FOR UPDATE GATE
            fc_layer = tf.reshape(fc_layer,[4,4,4,-1,128]) #[1,4,128,4,4]
            
            t_x_s_update = tf.nn.conv3d(prev_s,w[8],strides=[1,1,1,1,1],padding="SAME") + fc_layer
            t_x_s_reset = tf.nn.conv3d(prev_s,w[9],strides=[1,1,1,1,1],padding="SAME") + fc_layer

            update_gate = tf.sigmoid(t_x_s_update)

            complement_update_gate = 1 - update_gate
            reset_gate = tf.sigmoid(t_x_s_reset)

            rs = reset_gate * prev_s
            t_x_rs = tf.nn.conv3d(rs,w[10],strides=[1,1,1,1,1],padding="SAME") + fc_layer
            tanh_t_x_rs = tf.tanh(t_x_rs)
        
            gru_out = update_gate * prev_s + complement_update_gate * tanh_t_x_rs
            
            prev_s = gru_out
    
    with tf.name_scope("Decoder"):
        
        unpool7 = unpool(prev_s)
        conv7a = tf.nn.conv3d(unpool7,w_decoder[0],strides=[1,1,1,1,1],padding="SAME")
        conv7a = tf.nn.leaky_relu(conv7a,alpha=0.01)

        unpool8 = unpool(conv7a)
        conv8a = tf.nn.conv3d(unpool8,w_decoder[1],strides=[1,1,1,1,1],padding="SAME")
        conv8a = tf.nn.leaky_relu(conv8a,alpha=0.01)   

        unpool9 = unpool(conv8a)
        conv9a = tf.nn.conv3d(unpool9,w_decoder[2],strides=[1,1,1,1,1],padding="SAME")
        conv9a = tf.nn.leaky_relu(conv9a,alpha=0.01)

        conv10a = tf.nn.conv3d(conv9a,w_decoder[3],strides=[1,1,1,1,1],padding="SAME")
        conv10a = tf.nn.leaky_relu(conv10a,alpha=0.01)  

        conv11a = tf.nn.conv3d(conv10a,w_decoder[4],strides=[1,1,1,1,1],padding="SAME")
        ''' (32, 32, 32, 1, 2) '''

    #max_value = np.argmax(np.asarray(conv11a))
    #conv11a = conv11a - max_value
    #exp_x = tf.exp(conv11a)
    #sum_exp_x = tf.reduce_sum(exp_x,axis=4,keepdims=True) 
    sum_exp_x = tf.nn.softmax(conv11a,axis=-1)
    #''' (32, 32, 32, 1, 2) '''
    loss_ = loss(conv11a,sum_exp_x,Y)
    pred_ = sum_exp_x
    #pred_ = conv11a
    #For tensorboard
    tf.summary.scalar("loss", loss_)

    return loss_, pred_


def sample_loss(fc):
    fc = tf.nn.softmax(fc,axis=-1)
    l = Y - fc
    return tf.reduce_sum(l)


def loss(y,sum_exp_x,yhat):
    return tf.reduce_mean(
        tf.log(tf.nn.softmax(-yhat * y,axis=-1))+tf.log(sum_exp_x)
    )

def test_predict(pred,ind):
    pred_name = "test_pred_"+str(ind)+".obj"
    voxel.voxel2obj(pred_name,pred)

def unpool(x): #unpool_3d_zero_filled
    # https://github.com/tensorflow/tensorflow/issues/2169
    out = tf.concat([x, tf.zeros_like(x)], 2)
    out = tf.concat([out, tf.zeros_like(out)], 1)
    out = tf.concat([out, tf.zeros_like(out)], 0)

    sh = x.get_shape().as_list()
    out_size = [sh[0]*2, sh[1] * 2, sh[2] * 2, -1, sh[4]]
    return tf.reshape(out, out_size)

#loss_, pred_ = build_graph()
loss_ = try_encoder()

# Optimization
optimizer = tf.train.AdamOptimizer()
grads_vars = optimizer.compute_gradients(loss_)
clipped_gradients = []
for grad,var in grads_vars:
    if not grad is None:
          clipped_gradients.append((grad, var))
    #clipped_gradients.append(
    #    (tf.clip_by_value(grad,5.0,-5.0),var)) # 1 is max_gradient_norm)

updates = optimizer.apply_gradients(clipped_gradients)


if __name__=="__main__":
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:


        #sess = tf_debug.TensorBoardDebugWrapperSession(sess, "Berkan-MacBook-Pro.local:4334")
        # Run the initializer
        sess.run(init)

        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        summ = tf.summary.merge_all()

        train_writer = tf.summary.FileWriter('./train')
        train_writer.add_graph(sess.graph)
                                            
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

                ims = []

                for image in x_train[image_hash]:
                    ims.append(image)
                
                #ims = tf.convert_to_tensor(ims)
                #ims = tf.reshape(ims,[-1,127,127,3])
                #ims = ims.eval()
                #ims = np.random.rand(24,127,127,3)

                #vox = tf.convert_to_tensor(y_train[image_hash])
                #vox = tf.cast(vox,tf.float32)
                #vox = vox.eval()
                #batch_voxel = np.zeros((32,32,32,1,2), dtype=float)  
                #batch_voxel[:,:,:,0,0]= vox < 1
                #batch_voxel[:,:,:,0,1]= vox
                ims = np.random.rand(24,127,127,3)
                vox = np.ones((24,1024))
                                
                sess.run([updates], feed_dict={X: ims, Y: vox})


                l,s = sess.run([loss_, summ], feed_dict={X: ims, Y: vox})

                train_writer.add_summary(s, iter)
                
                print("OBJECT: " + str(iter)+" LOSS: "+str(l))
                with open("log.txt", "a") as myfile:
                    myfile.write("Iteration: "+str(iter)+" Loss: "+str(l)+"\n")
                #tf.summary.histogram('loss', forw[0])
                if iter % 5 == 0:
                    print("Testing Model at Iter ",iter)
                    print("HASH "+image_hash)
                    test_predict(o[:,:,:,0,1] > [0.4],iter)
                
                pbar.update(1)
            
            #x_train = dataset.train_data()
            #y_train = dataset.train_labels()

        pbar.close()
        print("Finished!")

