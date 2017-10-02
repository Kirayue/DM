import os, sys
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import random 
import time

def readImagefeature():
    #image_dict = np.load('test_image_feature.npy')
    image_dict = np.load('image_feature.npy')
    return image_dict

def readWord2vec():
    #word2vec_dict = np.load('test_google_vec_metadata.npy')
    word2vec_dict = np.load('google_vec_metadata.npy')
    return word2vec_dict

def readLabel():
    with open('./dataSets/t1_train_label.txt', 'r') as f:
        rawdata = f.read()
    rawdata = rawdata.split('\n')
    del rawdata[-1]
    return rawdata

def RNN(x, num_steps, n_hidden):
    x = tf.unstack(x, num_steps, 1)
    lstm_cell = rnn.LSTMCell(n_hidden, forget_bias=1.0, activation=tf.nn.relu, use_peepholes = True)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return outputs[-1]

def padding(length, arr):
    #print(length)
    #print(len(arr))
    if len(arr) == 0:
        arr = np.zeros((length, 300))
    elif len(arr) < length:
        for i in range(len(arr), length, 1):
            arr = np.append(arr, np.zeros((1, 300)), axis = 0)
    else:
        arr = arr[:length]
    #print(arr.shape)
    return arr

def FullyConnected(x, weights, biases, activation = None):
    Wx_plus_b = tf.matmul(x, weights) + biases
    if activation is None:
        outputs = Wx_plus_b
    else:
        outputs = activation(Wx_plus_b)
    return outputs


def getBatch(size, image_dict, metadata_dict, label, index):
    random.shuffle(index)
    y = []
    image = []
    title = []
    description = []
    keyword = []
    metadata = []
    for key in index[:size]:
    #for key in ['116731']:
        y.append([label[int(key)]])
        image.append(image_dict['{}.jpg'.format(key)][0][0][0])
        #print(np.array(metadata_dict[key]['title']).shape)
        title.append(padding(5, np.array(metadata_dict[key]['title'])))
        description.append(padding(10, np.array(metadata_dict[key]['description'])))
        keyword.append(padding(5, np.array(metadata_dict[key]['keyword'])))
        metadata.append([metadata_dict[key]['faveCount'], metadata_dict[key]['viewCount'], metadata_dict[key]['num_keyword'], metadata_dict[key]['commentCount'], metadata_dict[key]['num_gropus']])
        #print(key)
        #print(np.array(metadata_dict[key]['title']).shape)
        #print(np.array(metadata_dict[key]['description']).shape)
        #print(np.array(metadata_dict[key]['keyword']).shape)
    return np.array(image), np.array(title), np.array(description), np.array(keyword), np.array(metadata), np.array(y)

if __name__ == '__main__':
    image_dict = readImagefeature()[()]
    metadata_dict = readWord2vec()[()]
    #print(metadata_dict['1'].keys())
    image_key = set([key[:-4] for key in image_dict.keys()])
    metadata_key = set(metadata_dict.keys())
    inter_index = list(metadata_key.intersection(image_key))
    index = sorted(inter_index, key = lambda x: int(x))
    train_index = index[:int(len(index) * 0.8)]
    test_index = index[int(len(index) * 0.8):]
    label = readLabel()
    #print(image_key)
    #print('===================================')
    #print(metadata_key)
    #print(inter_index)

    ########### graph ############
    batch_size = 350
    display_step = 10
    learning_rate = 0.001
    iterations = 50000
    model_path = "./tensorflow_model/"
    log_path = './logs/'

    n_title = 100
    n_description = 150
    n_keyword = 100
    n_fc1 = 2048 + n_title + n_description + n_keyword + 5
    n_fc2 = 1000
    n_fc3 = 500
    n_output = 1

    with tf.name_scope('Input_Space'):
        title_feature = tf.placeholder(tf.float32, shape = (None, 5, 300), name = 'title')
        description_feature = tf.placeholder(tf.float32, shape = (None, 10, 300), name = 'description')
        keyword_feature = tf.placeholder(tf.float32, shape = (None, 5, 300), name = 'keyword')
        image_feature = tf.placeholder(tf.float32, shape = (None, 2048), name = 'image')
        metadata_feature = tf.placeholder(tf.float32, shape = (None, 5), name = 'metadata')
        y = tf.placeholder(tf.float32, shape = (None, 1), name = 'y')
    with tf.name_scope('Weights'):
        weights = {
                'fc1': tf.Variable(tf.random_normal([n_fc1, n_fc2])),
                'fc2': tf.Variable(tf.random_normal([n_fc2, n_fc3])),
                'fc3': tf.Variable(tf.random_normal([n_fc3, n_output])),
                }
    with tf.name_scope('biases'):
        biases = {
                'fc1': tf.Variable(tf.random_normal([n_fc2])),
                'fc2': tf.Variable(tf.random_normal([n_fc3])),
                'fc3': tf.Variable(tf.random_normal([n_output])),
                }

    with tf.variable_scope('Title_RNN'):
        vec_title = RNN(title_feature, 5, n_title)
    with tf.variable_scope('Description_RNN'):
        vec_description = RNN(description_feature, 10, n_description)
    with tf.variable_scope('keyword_RNN'):
        vec_keyword = RNN(keyword_feature, 5, n_keyword)
    with tf.name_scope('concate_features'):
        input_fc1 = tf.concat([image_feature, vec_title, vec_description, vec_keyword, metadata_feature], axis = 1)
    with tf.name_scope('fc1'):
        output_fc1 = FullyConnected(input_fc1, weights['fc1'], biases['fc1'] ,tf.nn.sigmoid)
    with tf.name_scope('fc2'):
        output_fc2 = FullyConnected(output_fc1, weights['fc2'], biases['fc2'] ,tf.nn.relu)
    with tf.name_scope('fc3'):
        pred = FullyConnected(output_fc2, weights['fc3'], biases['fc3'])
    
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
                image, title, description, keyword, metadata, batch_y = getBatch(batch_size, image_dict, metadata_dict, label, train_index)
                start = time.time()
                sess.run(optimizer, feed_dict = {title_feature: title, image_feature: image, description_feature: description, keyword_feature: keyword, metadata_feature: metadata, y: batch_y})
                end = time.time()
                print(end - start)
                if iters % display_step == 0:
                    mse, mae = sess.run([MSE, MAE], feed_dict = {title_feature: title, image_feature: image, description_feature: description, keyword_feature: keyword, metadata_feature: metadata, y: batch_y})
                    print("Iter: {}, Minibatch mse: {}, Minibatch mae: {}".format(iters, mse, mae))
                iters += 1
            print("Training done!")
            save_path = saver.save(sess, model_path + 'v1.ckpt')
            print('Model saved in file: ' + save_path)
            image, title, description, keyword, metadata, batch_y = getBatch(len(test_index), image_dict, metadata_dict, label, test_index)
            mse, mae = sess.run([MSE, MAE], feed_dict = {title_feature: title, image_feature: image, description_feature: description, keyword_feature: keyword, metadata_feature: metadata, y: batch_y})
            result = sess.run(pred, {title_feature: title, image_feature: image, description_feature: description, keyword_feature: keyword, metadata_feature: metadata, y: batch_y})
            ans = []
            for i, e in enumerate(result):
                ans.append("{}, {}".format(e[0], y[i][0]))
            with open('model_pred', 'w') as f:
                f.write('\n'.join(ans))
            print("Testing =>  Minibatch mse: {}, Minibatch mae: {}".format(mse, mae))
        else:
            print('HI')
            saver.restore(sess, model_path + 'v1.ckpt')
            print('Model restroed!')
            image, title, description, keyword, metadata, batch_y = getBatch(len(test_index), image_dict, metadata_dict, label, test_index)
            mse, mae = sess.run([MSE, MAE], feed_dict = {title_feature: title, image_feature: image, description_feature: description, keyword_feature: keyword, metadata_feature: metadata, y: batch_y})
            result = sess.run(pred, {title_feature: title, image_feature: image, description_feature: description, keyword_feature: keyword, metadata_feature: metadata, y: batch_y})
            print(result)
            ans = []
            for i, e in enumerate(result):
                ans.append("{}, {}".format(e[0], batch_y[i][0]))
            with open('model_pred', 'w') as f:
                f.write('\n'.join(ans))
            print("Testing =>  Minibatch mse: {}, Minibatch mae: {}".format(mse, mae))
    
