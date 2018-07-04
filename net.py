import tensorflow as tf
import voxel
import numpy as np
from PIL import Image

n_convfilter = [96, 128, 256, 256, 256, 256]
n_deconvfilter = [128, 128, 128, 64, 32, 2]
n_gru_vox = 4
n_fc_filters = [1024]

def initialize_weights():
    
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

    w = [w0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10,
     w11, w12, w13, w14, w15, w16, w17, w18, w19, w20,
     w21, w22, w23, w24, w25, w26, w27, w28, w29, w30, w31]

    return w





def predict(w,image,ind):
    im = np.array(Image.open(image))
    im = tf.convert_to_tensor(im)
    im = tf.reshape(im,[1,127,127,3])
    enc = encoder(w,im)
    tmp_state = tf.zeros_like(
        tf.truncated_normal([1,n_gru_vox,n_deconvfilter[0],n_gru_vox,n_gru_vox], stddev=0.5))
    g = gru(w,enc,tmp_state)
    dec = decoder(w,g)
    pred_name = "test_pred_"+str(ind)+".obj"
    res = dec.eval()
    voxel.voxel2obj(pred_name,res[0,:, :, :,1])




def train(w,x_train,y_train):
    print()
    '''Do nothing'''
    for images in x_train.keys():
        ims = [] # Concatenate N images
        for image in x_train[images]:
            #image = tf.convert_to_tensor(image) # (127,127,3)
            #image = tf.reshape(image,[3,127,127])
            ims.append(image) 

        # Initial empty GRU inputs
        prev_s = tf.Variable(tf.zeros_like(
            tf.truncated_normal([1,n_gru_vox,n_deconvfilter[0],n_gru_vox,n_gru_vox], stddev=0.5)), name="prev_s")
        
        tmp = encoder(w,ims)
        tmp = gru(w,tmp,prev_s)
        tmp = decoder(w,tmp)
        l = loss(tmp,y_train[images])
        tf.summary.histogram('loss', l)



def loss(x,y):
    y = tf.convert_to_tensor(y)
    y = tf.cast(y,tf.float32)
    l = tf.nn.softmax_cross_entropy_with_logits(logits=x[0,:,:,:,1],labels=y)
    '''TODO: Figure out:
        tf.nn.softmax_cross_entropy_with_logits 
            OR
        tf.nn.softmax_cross_entropy_with_logits_v2
    '''
    #return tf.metrics.mean(l)
    return tf.reduce_mean(l)




def encoder(w,ims):

    '''TODO: 
    Add leaky relus after each convolution layer
    '''
    
    # Input Layer
    ims = tf.convert_to_tensor(ims) # 24 Images are stored in a list, convert them to Tensor
    #X = ims #Placeholder
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
    ''' w[15] was [1024] , now its [1,1024]. Which one is correct?'''
    # [1,1024]

    return fc7


def gru(w,x_curr, prev_s):

    x_t = x_curr[0:1,:] # -> Take a single picture out of 24 pictures

    if(x_t.shape[0]==0): # Return output if images are finished.
        return prev_s
   
    #print("Iteration no:",x_curr.shape[0])

    ''' TODO : Broadcast-dot product or matmul ??'''
    fc_layer_U = tf.matmul(x_t,w[16]) #[1,1024]x[1024,8192] // FC LAYER FOR UPDATE GATE
    fc_layer_U = tf.reshape(fc_layer_U,[-1,4,128,4,4]) #[1,4,128,4,4]

    fc_layer_R = tf.matmul(x_t,w[31]) # FC LAYER FOR RESET GATE
    fc_layer_R = tf.reshape(fc_layer_R,[-1,4,128,4,4]) #[1,4,128,4,4]
    

    update_gate = tf.nn.conv3d(prev_s,w[17],strides=[1,1,1,1,1],padding="SAME")

    update_gate = update_gate + fc_layer_U
    update_gate = tf.sigmoid(update_gate)

    complement_update_gate = tf.subtract(tf.ones_like(update_gate),update_gate)

    reset_gate = tf.nn.conv3d(prev_s,w[18],strides=[1,1,1,1,1],padding="SAME")
    reset_gate = reset_gate + fc_layer_R
    reset_gate = tf.sigmoid(reset_gate)

    rs = tf.multiply(reset_gate,prev_s) # Element-wise multiply

    tanh_reset = tf.nn.conv3d(rs,w[19],strides=[1,1,1,1,1],padding="SAME")
    tanh_reset = tf.tanh(tanh_reset)

    gru_out = tf.add(
        tf.multiply(update_gate,prev_s),
        tf.multiply(complement_update_gate,tanh_reset)
    )
    #print("GRU OUT")
    #print(gru_out.shape)

    return gru(w,x_curr[1:,:],gru_out)

    
def decoder(w,s):
    
    s = tf.transpose(s,perm=[0,1,4,3,2]) # [(1, 4, 128, 4, 4)] -> [(1, 4, 4, 4, 128)]
    unpool7 = unpool(s)

    conv7a = tf.nn.conv3d(unpool7,w[20],strides=[1,1,1,1,1],padding="SAME")
    conv7a = tf.nn.leaky_relu(conv7a)

    conv7b = tf.nn.conv3d(conv7a,w[21],strides=[1,1,1,1,1],padding="SAME")
    conv7b = tf.nn.leaky_relu(conv7b)
    res7 = tf.add(unpool7,conv7b)

    unpool8 = unpool(res7)

    conv8a = tf.nn.conv3d(unpool8,w[22],strides=[1,1,1,1,1],padding="SAME")
    conv8a = tf.nn.leaky_relu(conv8a)   

    conv8b = tf.nn.conv3d(conv8a,w[23],strides=[1,1,1,1,1],padding="SAME")
    conv8b = tf.nn.leaky_relu(conv8b)    
    res8 = tf.add(unpool8,conv8b)

    unpool9 = unpool(res8)

    conv9a = tf.nn.conv3d(unpool9,w[24],strides=[1,1,1,1,1],padding="SAME")
    conv9a = tf.nn.leaky_relu(conv9a)   

    conv9b = tf.nn.conv3d(conv9a,w[25],strides=[1,1,1,1,1],padding="SAME")
    conv9b = tf.nn.leaky_relu(conv9b)  

    conv9c = tf.nn.conv3d(unpool9,w[26],strides=[1,1,1,1,1],padding="SAME")

    res9 = tf.add(conv9c,conv9b)

    conv10a = tf.nn.conv3d(res9,w[27],strides=[1,1,1,1,1],padding="SAME")
    conv10a = tf.nn.leaky_relu(conv10a)  
    
    conv10b = tf.nn.conv3d(conv10a,w[28],strides=[1,1,1,1,1],padding="SAME")
    conv10b = tf.nn.leaky_relu(conv10b)  

    conv10c = tf.nn.conv3d(conv10a,w[29],strides=[1,1,1,1,1],padding="SAME")
    conv10c = tf.nn.leaky_relu(conv10c)  

    res10 = tf.add(conv10c,conv10b)

    conv11a = tf.nn.conv3d(res10,w[30],strides=[1,1,1,1,1],padding="SAME")
    conv11a = tf.nn.leaky_relu(conv11a)  

    return conv11a


def unpool(value):
    """
    :param value: A Tensor of shape [b, d0, d1, ..., dn, ch]
    :return: A Tensor of shape [b, 2*d0, 2*d1, ..., 2*dn, ch]
    """
    #value = tf.transpose(value,perm=[0,1,3,4,2]) # [(1, 4, 4, 4, 128)]
    sh = value.get_shape().as_list()
    dim = len(sh[1:-1])
    out = (tf.reshape(value, [-1] + sh[-dim:]))
    for i in range(dim, 0, -1):
        out = tf.concat([out, tf.zeros_like(out)], i)
    out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
    out = tf.reshape(out, out_size)
    #out = tf.transpose(out,perm=[0,1,4,3,2]) # [(1, 4, 4, 4, 128)]
    return out