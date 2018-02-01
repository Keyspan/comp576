import tensorflow as tf 
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

learningRate = 1e-1
trainingIters = 60000
batchSize = 10
displayStep = 100


nInput = 28
nSteps = 28
nHidden = 128
nClasses = 10
acc_lstm = list()
acc_gru_64 = list()
acc_gru_32 = list()
loss_lstm = list()
loss_gru_64 = list()
loss_gru_32 = list()
acc_gru = list()
loss_gru = list()

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = {
	'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
	'out': tf.Variable(tf.random_normal([nClasses]))
}

def RNN(x, weights, biases):
    
    x = tf.transpose(x, [1,0,2])
    x = tf.reshape(x, [-1, nInput])
    x = tf.split(value = x, axis = 0, num_or_size_splits = nSteps) #configuring so you can get it as needed for the 28 pixels

    lstmCell = tf.contrib.rnn.LSTMCell(nHidden)#find which lstm to use in the documentation
    
    gruCell = tf.contrib.rnn.GRUCell(nHidden)

    outputs, states = tf.contrib.rnn.static_rnn(cell=gruCell,
                                                inputs = x,
                                                dtype = tf.float32
                                                )#for the rnn where to get the output and hidden state 

    return tf.matmul(outputs[-1], weights['out'])+ biases['out']

pred = RNN(x, weights, biases)

#optimization
#create the cost, optimization, evaluation, and accuracy
#for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.losses.softmax_cross_entropy(logits = pred, onehot_labels = y)

optimizer = tf.train.GradientDescentOptimizer (1e-1).minimize(cost)

correctPred = tf.equal(tf.argmax(y,1), tf.argmax(pred,1))

accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    
    sess.run(init)
    step = 1

    while step* batchSize < trainingIters:
        
        batchX, batchY = mnist.train.next_batch(batchSize)
        
        batchX = batchX.reshape((batchSize, nSteps, nInput))

        sess.run(optimizer, feed_dict={x:batchX, y:batchY})
        
        if step % displayStep == 0:
            acc = accuracy.eval(feed_dict={x:batchX, y:batchY})
            loss = cost.eval(feed_dict={x:batchX, y:batchY})
            print("Iter " + str(step*batchSize) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
            acc_gru.append(acc)
            loss_gru.append(loss)
        step +=1
    
    print('Optimization finished')

    testData = mnist.test.images.reshape((-1, nSteps, nInput))
    testLabel = mnist.test.labels
    gru_test_accuracy = sess.run(accuracy, feed_dict={x:testData, y:testLabel})
    print("Testing Accuracy:", gru_test_accuracy)
    
#plot
## accuracy and loss    
fig, ax = plt.subplots()
_, bx = plt.subplots()
ax.plot(range(len(acc_lstm)), acc_lstm, 'k', label='lstm trainig accuracy')
bx.plot(range(len(acc_lstm)), loss_lstm, 'k:', label='lstm trainig loss')
ax.plot(range(len(acc_lstm)), [lstm_test_accuracy]*len(acc_lstm),'k', 
        label = 'lstm test accuracy')
ax.plot(range(len(acc_lstm)), acc_gru, 'k--', label='gru trainig accuracy')
bx.plot(range(len(acc_lstm)), loss_gru, 'k--', label='gru trainig loss')
ax.plot(range(len(acc_lstm)), [gru_test_accuracy]*len(acc_lstm),'k--', 
        label = 'gru test accuracy')
ax.legend(loc='upper center', shadow=True, fontsize='x-large')
bx.legend(loc='upper center', shadow=True, fontsize='x-large')

##hidden units
_, cx = plt.subplots()
cx.plot(range(len(acc_gru)), acc_gru, 'k', label='gru trainig accuracy with 128 hidden units')
cx.plot(range(len(acc_gru)), acc_gru_64, 'k:', label='gru trainig accuracy with 64 hidden units')
cx.plot(range(len(acc_gru)), acc_gru_32, 'k--', label='gru trainig accuracy with 32 hidden units')
cx.plot(range(len(acc_gru)), [gru_test_accuracy]*len(acc_gru),'k', 
        label = 'gru test accuracy with 128')
cx.plot(range(len(acc_gru)), [gru_test_accuracy_64]*len(acc_gru),'k:', 
        label = 'gru test accuracy with 64')
cx.plot(range(len(acc_gru)), [gru_test_accuracy_32]*len(acc_gru),'k--', 
        label = 'gru test accuracy with 32')
cx.legend(loc='lower right', shadow=True)

plt.show()
