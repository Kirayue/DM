import numpy as np
from datetime import datetime,timedelta

def readFile(filename):
    with open(filename, 'r') as f: 
        rawData = f.read()
        rawData = rawData.split('\n')
        del rawData[-1]
    return rawData



if __name__ == '__main__':
    rawData = readFile('./t1_train_data.txt')
    rawLabel = readFile('./t1_train_label.txt')

