import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import json
from pprint import pprint
import 
def loadGloVe(filename):
    vocab = []
    embd = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            row = line.strip().split(' ')
            vocab.append(row[0])
            embd.append(row[1:])
            #print(row[1:])
            #exit()
    print("Loaded Glove!")
    return vocab, np.asarray(embd)

def getmetadata():
    with open('metadata', 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    filename = 'glove.6B.50d.txt'
    word_metadata = getmetadata()
    vec_metadata = {}
    vocab, embedding = loadGloVe(filename)
    vocab_size = len(vocab)
    embedding_dim = len(embedding[0])
    title_vocab_processor = learn.preprocessing.VocabularyProcessor(10)
    description_vocab_processor = learn.preprocessing.VocabularyProcessor(20)
    keyword_vocab_processor = learn.preprocessing.VocabularyProcessor(1)
    title_vocab_processor.fit(vocab)
    description_vocab_processor.fit(vocab)
    keyword_vocab_processor.fit(vocab)
    W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]), trainable=False, name="W")
    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
    embedding_init = W.assign(embedding_placeholder)
    with tf.Session() as sess:
        sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})
        count = 0
        for k, v in word_metadata.items():
            title = [v['title'].lower().strip()] if len(v['title'].lower().strip().split(' ')) > 0 else ['']
            #pprint(title)
            title = np.array(list(title_vocab_processor.transform(title)))
            #pprint(title)
            description = [v['description'].lower().strip()] if len(v['description'].lower().strip().split(' ')) > 0 else ['']
            description = np.array(list(description_vocab_processor.transform(description)))
            keyword = [ word.lower().strip() for word in v['keyword'] if word.lower().strip() in vocab]
            if len(keyword) > 5:
                keyword = keyword[:5]
            else:
                for i in range(len(keyword), 5, 1):
                    keyword.append('')
            keyword = np.array(list(keyword_vocab_processor.transform(keyword)))
            pprint('before: tf')
            vec_title = sess.run(tf.nn.embedding_lookup(W, title))
            vec_description = sess.run(tf.nn.embedding_lookup(W, description))
            vec_keyword = sess.run(tf.nn.embedding_lookup(W, keyword))
            pprint('before: ' + k)
            vec_metadata[k] = {
                    'num_gropus': v['num_groups'],
                    'commentCount': v['commentCount'],
                    'faveCount': v['faveCount'],
                    'num_keyword': len(v['keyword']),
                    'viewCount': v['viewCount'],
                    'title': vec_title, 
                    'description': vec_description, 
                    'keyword': vec_keyword 
                    }
            pprint('after: ' + str(k) + 'count: ' + str(count))
            count += 1
            #break
            if count % 100 == 0:
                np.save('vec_metadata.npy', vec_metadata)
        #print(sess.run(tf.nn.embedding_lookup(W, [[0, 2],[1, 3]])))



