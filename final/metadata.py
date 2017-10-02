import os, sys
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import json, random, time

def readmetadata():
    with open('metadata', 'r') as f:
        return json.load(f)

def readLabel():
    with open('./dataSets/t1_train_label.txt', 'r') as f:
        rawdata = f.read()
    rawdata = rawdata.split('\n')
    del rawdata[-1]
    return rawdata

def FullyConnected(x, weights, biases, activation = None):
    Wx_plus_b = tf.matmul(x, weights) + biases
    if activation is None:
        outputs = Wx_plus_b
    else:
        outputs = activation(Wx_plus_b)
    return outputs


def getBatch(size, label, metadata):
    temp = list(zip(metadata, label))
    random.shuffle(temp)
    x, y = zip(*temp)
    return np.array(x[:size]), np.reshape(np.array(y[:size]), (size, 1))

if __name__ == '__main__':
    metadata_dict = readmetadata()
    label = readLabel()
    y = []
    metadata = []
    for k in sorted(metadata_dict.keys(), key = lambda x: int(x)):
        y.append([label[int(k)]])
        metadata.append([metadata_dict[k]['faveCount'], metadata_dict[k]['viewCount'], len(metadata_dict[k]['keyword']), metadata_dict[k]['commentCount'], metadata_dict[k]['num_groups']])

    y = np.array(y)
    metadata = np.array(metadata)
    train_data = metadata[:int(len(metadata) * 0.8)]
    test_data = metadata[int(len(metadata) * 0.8):]
    train_y = y[:int(len(y) * 0.8)]
    test_y = y[int(len(y) * 0.8):]

    ########### graph ############
    batch_size = 130
    display_step = 10
    iterations = 10000
    learning_rate = 0.001
    model_path = "./tensorflow_model/"
    log_path = './logs/'

    n_fc1 = 5
    n_fc2 = 300
    n_fc3 = 100
    n_fc4 = 50
    n_output = 1

    with tf.name_scope('Input_Space'):
        metadata_feature = tf.placeholder(tf.float32, shape = (None, 5), name = 'metadata')
        y = tf.placeholder(tf.float32, shape = (None, 1), name = 'y')
    with tf.name_scope('Weights'):
        weights = {
                'fc1': tf.Variable(tf.random_normal([n_fc1, n_fc2])),
                'fc2': tf.Variable(tf.random_normal([n_fc2, n_fc3])),
                'fc3': tf.Variable(tf.random_normal([n_fc3, n_fc4])),
                'fc4': tf.Variable(tf.random_normal([n_fc4, n_output])),
                }
    with tf.name_scope('biases'):
        biases = {
                'fc1': tf.Variable(tf.random_normal([n_fc2])),
                'fc2': tf.Variable(tf.random_normal([n_fc3])),
                'fc3': tf.Variable(tf.random_normal([n_fc4])),
                'fc4': tf.Variable(tf.random_normal([n_output])),
                }

    with tf.name_scope('fc1'):
        output_fc1 = FullyConnected(metadata_feature, weights['fc1'], biases['fc1'] ,tf.nn.sigmoid)
    with tf.name_scope('fc2'):
        output_fc2 = FullyConnected(output_fc1, weights['fc2'], biases['fc2'] ,tf.nn.relu)
    with tf.name_scope('fc3'):
        output_fc3 = FullyConnected(output_fc2, weights['fc3'], biases['fc3'] ,tf.nn.sigmoid)
    with tf.name_scope('fc3'):
        pred = FullyConnected(output_fc3, weights['fc4'], biases['fc4'])
    
    with tf.name_scope('Cost'):
        cost = tf.reduce_mean(tf.pow((y - pred), 2))
    with tf.name_scope('Evaluation'):
        MAE = tf.reduce_mean(tf.abs(pred - y))
        MSE = cost 
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        if sys.argv[1] == 'train':
            sess.run(init)
            writer = tf.summary.FileWriter(log_path, graph = sess.graph)
            iters = 1
            while iters < iterations:
                start = time.time()
                batch_x, batch_y = getBatch(batch_size, train_y, train_data)
                sess.run(optimizer, feed_dict = {metadata_feature: batch_x, y: batch_y})
                end = time.time()
                print(end - start)
                if iters % display_step == 0:
                    mse, mae = sess.run([MSE, MAE], feed_dict = {metadata_feature: batch_x, y: batch_y})
                    print("Iter: {}, Minibatch mse: {}, Minibatch mae: {}".format(iters, mse, mae))
                iters += 1
            print("Training done!")
            save_path = saver.save(sess, model_path + 'metadata_v1.ckpt')
            print('Model saved in file: ' + save_path)
            mse, mae = sess.run([MSE, MAE], feed_dict = {metadata_feature: test_data, y: test_y})
            result = sess.run(pred, feed_dict = {metadata_feature: test_data, y: test_y})
            ans = []
            for i, e in enumerate(result):
                ans.append("{}, {}".format(e[0], test_y[i][0]))
            with open('matadata_model_pred', 'w') as f:
                f.write('\n'.join(ans))
            print("Testing =>  Minibatch mse: {}, Minibatch mae: {}".format(mse, mae))
        else:
            saver.restore(sess, model_path + 'metadata_v1.ckpt')
            print('Model restroed!')
    
