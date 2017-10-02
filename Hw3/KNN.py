import numpy as np
import sys
from collections import Counter
def readData(filename):
    with open(filename, 'r') as f:
        rawdata = f.read()
    return rawdata
def preprocessData(rawdata):
    rawdata = rawdata.replace('Basic', '1')
    rawdata = rawdata.replace('Normal', '2')
    rawdata = rawdata.replace('Silver', '3')
    rawdata = rawdata.replace('Gold', '4')
    rawdata = rawdata.replace('S', '1')
    rawdata = rawdata.replace('M', '2')
    rawdata = rawdata.split('\n')
    rawdata.pop()
    data = np.zeros([len(rawdata), 5], dtype=np.int64)
    for index, rowdata in enumerate(rawdata):
        rowdata = rowdata[1:-1].split(',')
        tmp = {}
        for attrStr in rowdata:
            attr = attrStr.split()
            tmp[attr[0]] = int(attr[1])
        data[index][0] = tmp['0'] if '0' in tmp else 0 
        data[index][1] = tmp['1'] if '1' in tmp else 0
        data[index][2] = tmp['3'] if '3' in tmp else 0
        data[index][3] = tmp['4'] if '4' in tmp else 0
        data[index][4] = tmp['2'] if '2' in tmp else 1
    return data
def similarity(train, target):
    dotResult = np.matmul(train, target)
    normOfTrain = np.linalg.norm(train, axis=1)
    normOfTarget = np.linalg.norm(target)
    cosineSimilarity = dotResult / (normOfTrain * normOfTarget)
    return cosineSimilarity
def toMembercard(num):
    if num is 1:
        return 'Basic'
    elif num is 2:
        return 'Normal'
    elif num is 3:
        return 'Silver'
    else:
        return 'Gold'
def acc(pred, ans):
    count = 0
    for index, value in enumerate(ans):
        if value == pred[index]:
            count += 1
    return count / len(ans) * 100
if __name__ == '__main__':
    K = int(sys.argv[2])
    train_rawdata = readData('training')
    test_rawdata = readData('test')
    train_data = preprocessData(train_rawdata)
    test_data = preprocessData(test_rawdata)
    test_rawdata = test_rawdata.split('\n')
    test_rawdata.pop()
    pred = []
    with open('output.txt','w') as f:
        for index, value in enumerate(test_rawdata):
            cosine = similarity(train_data[:,:4], test_data[index,:4] )
            sortindex = sorted(range(len(train_data)), key = lambda k:cosine[k], reverse=True)
            c = Counter([train_data[i][4] for i in sortindex][:K])
            f.write(value + ' member_card = ' + toMembercard(int(c.most_common(1)[0][0])) + '\n')
            pred.append(int(c.most_common(1)[0][0]))
        result = acc(pred, test_data[:, 4])
        f.write('Accurancy = ' + str(result) + ' with K = ' + str(K)) 
    print('Accurancy = ' + str(result) + ' with K = ' + str(K))
