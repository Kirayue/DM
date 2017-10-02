import sys, os
import json
from pprint import pprint
import math
def preprocessmetadata():
    metadataList = ['metadata_1', 'metadata_2', 'metadata_4', 'metadata_5', 'metadata_6']
    data = {}
    for metadatafile in metadataList:    
        with open(metadatafile, 'r') as f:
            data.update(json.load(f))
    #pprint(data)
    with open('metadata', 'w') as f:
        json.dump(data, f)

def getmetadata():
    with open('metadata', 'r') as f:
        return json.load(f)

def readlabel():
    with open('./dataSets/t1_train_label.txt', 'r') as f:
        rawdata = f.read()
    rawdata = rawdata.split('\n')
    del rawdata[-1]
    return rawdata

def produceSVMdata(label, metadata):
    data = []
    #count = 0
    for k in sorted(metadata.keys(), key = lambda x: int(x)):
        print(k)
        v = metadata[k]
        #print(v['title'].split(' '))
        #exit()
        rowdata = '{} 1:{} 2:{} 3:{} 4:{} 5:{}'.format(label[int(k)], v['viewCount'], len(v['keyword']), v['num_groups'], v['faveCount'], v['commentCount'])
        #print(rowdata)
        data.append(rowdata)
        #count += 1
        #if count == 11:
            #break

    with open('./SVM/train_data', 'w') as f:
        f.write('\n'.join(data[:int(len(data) * 0.8)]))
    with open('./SVM/test_data', 'w') as f:
        f.write('\n'.join(data[int(len(data) * 0.8):]))


if __name__ == '__main__':
    #preprocessmetadata()
    label = readlabel()
    metadata = getmetadata()
    produceSVMdata(label, metadata)
