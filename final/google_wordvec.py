import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import json
from pprint import pprint
import gensim
def getmetadata():
    with open('metadata', 'r') as f:
        return json.load(f)

if __name__ == '__main__':
    wv = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True) 
    word_metadata = getmetadata()
    vec_metadata = {}
    vocab = wv.vocab.keys()
    vocab_size = len(vocab)
    embedding_dim = 300
    count = 0
    word = ['123563', '256525', '265795', '362383', '42274', '331942', '185055', '144630', '364016', '116731', '209071', '217205', '25260', '295970', '209372', '231992', '24633', '1', '175011', '246388', '80247', '93638', '154461', '56437', '31446', '308441', '232207', '118455', '232550', '373762', '139754', '122930', '25261', '34935', '201961', '204917', '158031', '184927', '83058', '366070', '147504', '157588', '196022', '263130', '298986', '230242', '43369', '355552', '264153', '14440', '67012', '214523', '330100', '337891', '177014', '36148', '149460', '346728', '194264', '373633', '66420', '69652', '238905', '135963', '148992', '175091', '149065', '321293', '256984', '173058', '32601', '160625', '30102', '373164', '296389', '212830', '191213', '181855', '369816', '342651', '1567', '121731', '97964', '287763', '268784', '253198', '150235', '211345', '293168', '63687', '83909', '97765', '210013', '167381', '103816', '262034', '104464', '110447', '285390', '58105']

    #for k, v in word_metadata.items():
    for k in word:
        try:
            v = word_metadata[k]
        except:
            print("NO such key: " + k)
            continue
        #k = '123563'
        #print(word_metadata[k]['title'])
        title = [wv[word] for word in v['title'].lower().strip().split(' ') if word in vocab] if len(v['title'].lower().strip().split(' ')) > 0 else []
        #exit()
        description = [wv[word] for word in v['description'].lower().strip().split(' ') if word in vocab] if len(v['description'].lower().strip().split(' ')) > 0 else []
        keyword = [wv[word.lower().strip()] for word in v['keyword'] if word.lower().strip() in vocab]
        vec_metadata[k] = {
                'num_gropus': v['num_groups'],
                'commentCount': v['commentCount'],
                'faveCount': v['faveCount'],
                'num_keyword': len(v['keyword']),
                'viewCount': v['viewCount'],
                'title': title, 
                'description': description, 
                'keyword': keyword 
                }
        print(k)
        pprint('Count: ' + str(count))
        count += 1
        #print(vec_metadata[k])
        #if count % 100 == 0:
        #    np.save('test_google_vec_metadata.npy', vec_metadata)
        #    print('save!')
        #    break
    np.save('test_google_vec_metadata.npy', vec_metadata)
    print('save~~~!')
