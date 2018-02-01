from scipy import misc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# --------------------------------------------------
# setup

def weight_variable(shape, name):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    initial = tf.contrib.layers.xavier_initializer()
    W = tf.get_variable(name = name, shape = shape, initializer = initial)
    return W

def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    initial = tf.constant(0.1, shape=shape)
    b = tf.Variable(initial)
    return b

def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    # IMPLEMENT YOUR CONV2D HERE
    h_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    return h_conv

def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    h_max = tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME')
    return h_max


ntrain =  500 # per class
ntest =  88 # per class
nclass =  10 # number of classes
imsize = 28
nchannels = 1
batchsize = 50

Train = np.zeros((ntrain*nclass,imsize,imsize,nchannels))
Test = np.zeros((ntest*nclass,imsize,imsize,nchannels))
LTrain = np.zeros((ntrain*nclass,nclass))
LTest = np.zeros((ntest*nclass,nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, ntrain):
        #path = '/Users/py/Python/comp_576/CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
        path = '/home/py/CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itrain += 1
        Train[itrain,:,:,0] = im
        LTrain[itrain,iclass] = 1 # 1-hot lable
    for isample in range(0, ntest):
        #path = '/Users/py/Python/comp_576//CIFAR10/Test/%d/Image%05d.png' % (iclass,isample)
        path = '/home/py/CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
        im = misc.imread(path); # 28 by 28
        im = im.astype(float)/255
        itest += 1
        Test[itest,:,:,0] = im
        LTest[itest,iclass] = 1 # 1-hot lable

sess = tf.InteractiveSession()

tf_data = tf.placeholder(tf.float32, shape=[None, imsize, imsize, nchannels])
tf_labels = tf.placeholder(tf.float32, shape=[None, nclass])#tf variable for labels

# --------------------------------------------------
# model
#create your model

# first convolutional layer and max pooling layer
W_conv1 = weight_variable([5, 5, 1, 32], 'W_conv1')
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.tanh(conv2d(tf_data, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer and max pooling layer
W_conv2 = weight_variable([5, 5, 32, 64], 'W_conv2')
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.tanh(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# first fully connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024], 'W_fc1')
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.tanh(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# second fully connected layer
W_fc2 = weight_variable([1024, 10], 'W_fc2')
b_fc2 = bias_variable([10])
h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    
# --------------------------------------------------
# loss
#set up the loss, optimization, evaluation, and accuracy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_labels, logits=h_fc2))
optimizer = tf.train.AdagradOptimizer(learning_rate = 1e-2).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(h_fc2, 1), tf.argmax(tf_labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# --------------------------------------------------
# optimization
sess.run(tf.global_variables_initializer())
batch_xs = np.zeros((batchsize, imsize, imsize, nchannels))#setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_ys = np.zeros((batchsize, nclass))#setup as [batchsize, the how many classes] 
adagrad_losses = list()
adagrad_accs = list()
def train():
    for i in range(5000): # try a small iteration size once it works then continue
        perm = np.arange(ntrain*nclass)
        np.random.shuffle(perm)
        feed = {tf_data:batch_xs, tf_labels:batch_ys, keep_prob: 0.5}
        for j in range(batchsize):
            batch_xs[j,:,:,:] = Train[perm[j],:,:,:]
            batch_ys[j,:] = LTrain[perm[j],:]
        loss = cross_entropy.eval(feed_dict = feed)
        adagrad_losses.append(loss)
        acc = accuracy.eval(feed_dict = feed)
        first_weight = W_conv1.eval()
        adagrad_accs.append(acc)
        if i%100 == 0:
            print('{}th step, loss is {}, train accuracy is {}'.format(i,loss,acc))#calculate train accuracy and print it
        optimizer.run(feed_dict={tf_data:batch_xs, tf_labels:batch_ys, keep_prob: 0.5}) # dropout only during training

    # --------------------------------------------------
    # test
    act1 = h_conv1.eval(feed_dict = {tf_data: Test, tf_labels: LTest, keep_prob: 1.0})
    act2 = h_conv2.eval(feed_dict = {tf_data: Test, tf_labels: LTest, keep_prob: 1.0})
    act3 = h_fc1.eval(feed_dict = {tf_data: Test, tf_labels: LTest, keep_prob: 1.0})
    act4 = h_fc2.eval(feed_dict = {tf_data: Test, tf_labels: LTest, keep_prob: 1.0})
    print("test accuracy %g"%accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))
    
    return adagrad_losses, adagrad_accs, first_weight, act1,act2,act3,act4

# plot loss iteration with different learning rate
_, ax = plt.subplots()

ax.plot(range(100), lr2_accs, 'k', label='accuracy for 1e-2 learning rate')
ax.plot(range(100), lr3_accs, 'k:', label='accuracy for 1e-3 learning rate')
ax.plot(range(100), lr4_accs, 'k--', label='accuracy for 1e-4 learning rate')
ax.legend(loc='upper right', shadow=True)

plt.show()

# plot loss iteration with different optimizer
_, bx = plt.subplots()
bx.plot(range(100), adagrad_losses, 'k', label='Adagrad Optimizer')
bx.plot(range(100), grad_losses, 'k:', label='Gradient Descent')
bx.plot(range(100), mom_losses, 'k--', label='Momentum')
bx.legend(loc='upper right', shadow=True)

plt.show()

# visualize weight
_, cx = plt.subplots()
cx.plot(range(10000),acc, 'k', label  = 'accuracy for 10000 iterations')
cx.legend(loc = 'upper right', shadow = True)
plt.show()

# import weight data 
weight = np.load('/Users/py/Python/comp_576/hw2/weight.npy')
weight_squeeze = np.squeeze(weight)
weight_image = np.transpose(weight_squeeze,[2,0,1])
    
fig = plt.figure()
for i in range(32):
    ax = fig.add_subplot(4, 8, 1 + i)
    ax.imshow(weight_image[i,:,:],cmap='gray')
    plt.axis('off')

ac1 = np.load('/Users/py/Python/comp_576/hw2/ac1.npy')
ac2 = np.load('/Users/py/Python/comp_576/hw2/ac2.npy')

# visualize activation
np.argmax(ac1)
np.max(ac1)
invest1 = ac1[52,:,:,4]
np.mean(invest1>0)

np.argmax(ac2)
np.max(ac2)
invest2 = ac2[727,0,11,39]
np.mean(invest2>0)
