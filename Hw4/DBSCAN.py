import numpy as np
import sys
from numpy import linalg as LA
from sklearn import preprocessing
def readData(filename):
    with open(filename, 'r') as f:
        rawdata = f.read()
    return rawdata

def preprocessData(rawdata):
    rawdata = rawdata.split('\n')
    rawdata.pop()
    data = []
    for index, rowdata in enumerate(rawdata):
        rawdata[index] = np.array(list(map(float, (rowdata.split()))))
    nor_data = np.array(rawdata)
    nor_data = nor_data / nor_data.max(axis=0)
    for index, rowdata in enumerate(nor_data):
        data.append({
            'attr': rowdata,
            'visited': False,
            'clusterID': "",
            'rawdata': rawdata[index]
            })
    return data

def getNeighbors(radius, P, datas):
    neighbor = [data for data in datas if LA.norm(P['attr'] - data['attr']) < radius]
    return neighbor

def expandCluster(P, neighbors, label, radius, minPts, D):
    P['clusterID'] = label
    for rowdata in neighbors:
        if not rowdata['visited']:
            rowdata['visited'] = True
            neighbor = getNeighbors(radius, rowdata, D)
            if len(neighbor) >= minPts:
                neighbors.extend(neighbor)
            if not rowdata['clusterID']:
                rowdata['clusterID'] = label

if __name__ == '__main__':
    rawdata = readData(sys.argv[2])
    datas = preprocessData(rawdata)
    radius = float(sys.argv[4])
    minPts = float(sys.argv[6])
    label = 0
    for data in datas:
        if not data['visited']:
            data['visited'] = True
            neighbor = getNeighbors(radius, data, datas)
            if len(neighbor) < minPts:
                data['clusterID'] = "Noise"
            else:
                label += 1
                expandCluster(data, neighbor, label, radius, minPts, datas)
    print('There are ' + str(label) + ' clusters.')
    with open(sys.argv[8], 'w') as f:
        for data in datas:
            f.write(str(data['rawdata'][0]) + "  " + str(data['rawdata'][1]) + "  " + str(data['clusterID']) + '\n')
